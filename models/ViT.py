import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms, datasets
import random
import numpy as np
from tqdm import tqdm
import torchvision.models as models
import timm

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


    for epoch in range(num_epochs):
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

class ViT(nn.Module):
    def __init__(self, num_classes=2):
        super(ViT, self).__init__()
        # Load the pre-trained ViT model
        self.ViT = timm.create_model('vit_base_patch16_224', pretrained=False)
        
        # Modify the final fully connected layer for the desired number of output classes
        self.ViT.head = nn.Linear(self.ViT.head.in_features, num_classes)

    def forward(self, x):
        x = self.ViT(x)
        return x
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
    batch_size = 32
    num_epochs = 10

    # Create training and validation datasets
    #data_dir = 'datasets_old/png_images/' #update this path
    #data_dir = 'datasets/png_images'
    data_dir = '/home/alcor/students/alexandraCDDB/faces'
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
    model = ViT().to(device)

    # Define the loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)


    # Create a folder to save your model
    model_name = 'ViT_scratch-gan'
    create_dir(model_name)

    # Train and evaluate your model
    model, hist, best_acc = train_model(model, model_name, dataloaders_dict, criterion, optimizer, num_epochs=num_epochs)

    # Evaluate the model on the test set
    hist["test_acc"] = evaluate(model, dataloaders_dict['test'])

    # Save training and evaluation history
    save_all(model_name, hist)

if __name__ == '__main__':
    main()



