import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms, datasets
import random
import numpy as np
from tqdm import tqdm
import torchvision.models as models

# Set random seeds for reproducibility
random.seed(0)
np.random.seed(0)
torch.manual_seed(0)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


# Detect if we have a GPU available
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def evaluate(model, testloader):
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in testloader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return correct / total


def train_model(model, model_name, dataloaders, criterion,optimizer, num_epochs=25):

    history = {}
    history['val_acc'] = []
    history['train_acc'] = []
    best_acc = 0.0


    for epoch in tqdm(range(num_epochs)):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))

        # Each epoch has a training and validation phase
        for phase in ['train', 'eval']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    # Get model outputs and calculate loss
                    # Special case for inception because in training it has an auxiliary output. In train
                    #   mode we calculate the loss by summing the final output and the auxiliary output
                    #   but in testing we only consider the final output.
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)

                    _, preds = torch.max(outputs, 1)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            epoch_acc = running_corrects.double() / len(dataloaders[phase].dataset)

            lr = optimizer.param_groups[0]['lr']

            if phase == 'eval':
                print("val_acc: ", epoch_acc.item())
                history['val_acc'].append(epoch_acc)
                if epoch_acc > best_acc:
                    best_acc = epoch_acc
                    torch.save(model, model_name + '/'  + "best.pth")
               
            else:
                print("train_acc: ", epoch_acc.item())
                history['train_acc'].append(epoch_acc)

        print(f'model name: {model_name}, learning rate: {lr}')
        print('-' * 10)

    return model, history, best_acc

# Custom ResNet without in-place operations
from torchvision.models.resnet import BasicBlock, ResNet

class CustomResNet(ResNet):
    def __init__(self, block, layers, num_classes=1000, zero_init_residual=False,
                 groups=1, width_per_group=64, replace_stride_with_dilation=None,
                 norm_layer=None):
        super(CustomResNet, self).__init__(block, layers, num_classes, zero_init_residual,
                                           groups, width_per_group, replace_stride_with_dilation,
                                           norm_layer)
        # Change all ReLU in-place operations to non in-place
        for m in self.modules():
            if isinstance(m, nn.ReLU):
                m.inplace = False

def resnet18(pretrained=False, progress=True, **kwargs):
    return CustomResNet(BasicBlock, [2, 2, 2, 2], **kwargs)

# Hybrid CNN-Transformer Model

"""
class CNNTransformer(nn.Module):
    def __init__(self, num_classes=2, num_transformer_layers=4):
        super(CNNTransformer, self).__init__()
        self.backbone = resnet18(pretrained=True)
        self.backbone = nn.Sequential(*list(self.backbone.children())[:-1])
       
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=512, nhead=8), num_layers=num_transformer_layers)
       
        self.classifier = nn.Linear(512, num_classes)

    def forward(self, x):
        x = self.backbone(x)
        x = torch.flatten(x, start_dim=2)
        x = x.permute(2, 0, 1)
        x = self.transformer(x)
        x = x.mean(dim=0)
        x = self.classifier(x)
        return x
"""
    
    ######################################################################################################
    ##################### ANOTHER VERSION ##############################################################

    # Define the Backbone class with ResNet-50

class BackboneResNet50(nn.Module):
    def __init__(self):
        super(BackboneResNet50, self).__init__()
        self.backbone = models.resnet50(pretrained=True)
        self.backbone.fc = nn.Identity()

    def forward(self, x):
        x = self.backbone(x)
        return x



# Modify the CNNTransformer class to use the BackboneResNet50
class CNNTransformer(nn.Module):
    def __init__(self, num_classes=2, num_transformer_layers=4, hidden_dim=1024):
        super(CNNTransformer, self).__init__()

        # Use the BackboneResNet50 as the backbone
        self.backbone = BackboneResNet50()

        # Define the transformer encoder with more heads
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(hidden_dim ,nhead=8), num_layers=num_transformer_layers
        )

        # Linear classifier
        self.classifier = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        # Pass input through the backbone
        x = self.backbone(x)

        # Check the shape of x and adjust it if needed
        if x.dim() > 2:
            x = torch.flatten(x, start_dim=2)
            x = x.permute(2, 0, 1)
        else:
            # If x has only two dimensions, reshape it to have batch dimension
            x = x.unsqueeze(0)

        # Apply the transformer encoder
        x = self.transformer(x)

        # Compute the mean over time steps (sequence length)
        x = x.mean(dim=0)

        # Pass through the classifier
        x = self.classifier(x)

        return x
    
#########################################################################################################################
#########################################################################################################################

def save_all(model_name, history):
    
    path_train = model_name + '/train.txt'
    path_eval = model_name + '/eval.txt'
    path_test = model_name + '/test.txt'
    for i, path in enumerate([path_train,path_eval,path_test]):
        with open(path, 'w') as f:
            if i == 0:
                ohist = [h.cpu().numpy() for h in history['train_acc']]
                for el in ohist:
                    f.write(str(el)+'\n')
            elif i == 1:
                ohist = [h.cpu().numpy() for h in history['val_acc']]
                for el in ohist:
                    f.write(str(el)+'\n')
            else:
                el = history['test_acc']
                f.write(str(el)+'\n')

def create_dir(path):
    if not os.path.exists(path):
        os.mkdir(path)
        print(f'created folder: {path}')
    else:
        print(f'directory {path} already exists')

def get_data_transform(input_size):
    transform = transforms.Resize((input_size, input_size))

    data_transforms = {
        'train': transforms.Compose([
            transform,
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'eval': transforms.Compose([
            transform,
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'test': transforms.Compose([
        transform,
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    }

    return data_transforms

# Define custom dataset loader to balance real and fake images in training
class CustomBalancedImageFolder(datasets.ImageFolder):
    def __init__(self, root, transform=None):
        super(CustomBalancedImageFolder, self).__init__(root, transform=transform)
        self.fake_samples = [x for x in self.samples if x[1] == 0]
        self.real_samples = [x for x in self.samples if x[1] == 1]
        min_samples = min(len(self.real_samples), len(self.fake_samples))
        self.samples = self.real_samples[:min_samples] + self.fake_samples[:min_samples]



def main():
    # Parameters
    input_size = 224
    workers = 6
    batch_size = 64
    num_epochs = 10

    # Create training and validation datasets
    data_dir = '/RealFaces_w_StableDiffusion/CDDB/faces'
    train_dataset = CustomBalancedImageFolder(os.path.join(data_dir, 'train'), transform=get_data_transform(input_size)['train'])
    eval_dataset = CustomBalancedImageFolder(os.path.join(data_dir, 'eval'), transform=get_data_transform(input_size)['eval'])
    test_dataset = CustomBalancedImageFolder(os.path.join(data_dir, 'test'), transform=get_data_transform(input_size)['test'])


    # Create training and validation dataloaders
    dataloaders_dict = {
        'train': torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=workers),
        'eval': torch.utils.data.DataLoader(eval_dataset, batch_size=batch_size, shuffle=True, num_workers=workers),
        'test': torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=True, num_workers=workers)
    }

    # Initialize your custom CNN-Transformer model
    #model = CNNTransformer().to(device)
    model = CNNTransformer(num_classes=2, num_transformer_layers=1, hidden_dim=2048).to(device)

    # Define the loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

    # Create a folder to save your model
    model_name = 'hybrid_gan_data_new'
    path = '/RealFaces_w_StableDiffusion/'
    create_dir(os.path.join(path, model_name))


    # Train and evaluate your model
    model, hist, best_acc = train_model(model, model_name, dataloaders_dict, criterion, optimizer, num_epochs=num_epochs)
    

    # Evaluate the model on the test set
    hist["test_acc"] = evaluate(model, dataloaders_dict['test'])

    # Save training and evaluation history
    save_all(model_name, hist)

if __name__ == '__main__':
    main()
