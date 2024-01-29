import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms, datasets
import random
import numpy as np
from tqdm import tqdm
import math 

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

    confidence = []
    total_confidence = 0
    var_confidence = 0
    t_confidence = [] 
    f_confidence = []
    true_confidence = 0
    false_confidence = 0
    true_var_conf = 0
    false_var_conf = 0
    fake_correct = 0
    real_correct = 0

    total_fake = 0
    total_real = 0

    with torch.no_grad():
        for inputs, labels in testloader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)

            #statistics
            probs = torch.nn.functional.softmax(outputs,dim=1)
            probs = probs*5
            conf,classes = torch.max(probs,1)
            
            confidence += list(conf.cpu().numpy())
            t_confidence += list(conf[classes == 0].cpu().numpy())
            f_confidence += list(conf[classes == 1].cpu().numpy())


            total_fake += labels[labels == 1].size(0)
            total_real += labels[labels == 0].size(0)

            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            fake_correct += (predicted[labels ==1] == labels[labels == 1]).sum().item()
            real_correct += (predicted[labels ==0] == labels[labels == 0]).sum().item()
            

    total_confidence = np.mean(confidence)
    var_confidence = np.var(confidence)
    true_confidence = np.mean(t_confidence)
    false_confidence = np.mean(f_confidence)
    true_var_conf = np.var(t_confidence)
    false_var_conf = np.var(f_confidence)
          
    return correct / total , total_confidence ,var_confidence ,true_confidence ,false_confidence ,true_var_conf , false_var_conf, fake_correct, real_correct


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

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=0.1)
        
        # Create the positional encodings matrix
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        
        self.register_buffer('pe', pe)
        
    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)

# Hybrid CNN-Transformer Model
class CNNTransformer(nn.Module):
    def __init__(self, num_classes=2, num_transformer_layers=4):
        super(CNNTransformer, self).__init__()
        self.backbone = resnet18(pretrained=False)
        self.backbone = nn.Sequential(*list(self.backbone.children())[:-1])
        self.positional_encoding = PositionalEncoding(d_model=512)
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=512, nhead=8), num_layers=num_transformer_layers)
        self.classifier = nn.Linear(512, num_classes)

    def forward(self, x):
        
        x = self.backbone(x)
        x = torch.flatten(x, start_dim=2)
        x = x.permute(2, 0, 1)
        x = self.positional_encoding(x)  # Apply positional encoding
        x = self.transformer(x)
        x = x.mean(dim=0)
        x = self.classifier(x)
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
                f.write('test_confidence '+ str(history['test_confidence'])+'\n')
                f.write('test_var_confidence '+ str(history['test_var_confidence'])+'\n') 
                f.write('test_true_confidence '+ str(history['test_true_confidence'])+'\n') 
                f.write('test_false_confdence ' + str(history['test_false_confdence'] )+'\n') 
                f.write('test_true_var_conf ' + str(history['test_true_var_conf'])+'\n') 
                f.write('test_false_var_conf ' + str(history['test_false_var_conf'] )+'\n') 
                f.write('test_fake_correct '+ str(history['test_correct'] )+'\n') 
                f.write('test_real_correct' + str(history['test_real_correct'] )+'\n') 



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
    data_dir = '/RealFaces_w_StableDiffusion/datasets_old/png_images/' # Update this path
    #data_dir = '/RealFaces_w_StableDiffusion/CDDB/faces' #Gan data for generalization test

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
    model = CNNTransformer().to(device)

    # Define the loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

    # Create a folder to save your model
    model_name = 'hybrid_PosEncoding_scratch'
    create_dir(model_name)

    # Train and evaluate your model
    model, hist, best_acc = train_model(model, model_name, dataloaders_dict, criterion, optimizer, num_epochs=num_epochs)

    acc , confidence ,var_confidence ,true_confidence ,false_confidence ,true_var_conf , false_var_conf, fake_correct, real_correct = evaluate(model,dataloaders_dict['test'])
    
    
    # Evaluate the model on the test set
    hist["test_acc"] = acc

    hist['test_confidence'] = confidence
    hist['test_var_confidence'] = var_confidence
    hist['test_true_confidence']= true_confidence
    hist['test_false_confdence'] = false_confidence
    hist['test_true_var_conf'] = true_var_conf
    hist['test_false_var_conf'] = false_var_conf
    hist['test_fake_correct'] = fake_correct
    hist['test_real_correct'] = real_correct

    # Save training and evaluation history
    save_all(model_name, hist)

if __name__ == '__main__':
    main()


