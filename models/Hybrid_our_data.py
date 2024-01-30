import os
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader, Dataset, random_split
from PIL import Image
import random
import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np

# Set random seeds for reproducibility
random.seed(0)
np.random.seed(0)
torch.manual_seed(0)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

"""
The model is a hybrid CNN-Transformer architecture designed for image classification tasks.


Backbone CNN (ResNet-18): The model starts with a ResNet-18 convolutional neural network (CNN) backbone.
ResNet-18 is a deep CNN architecture known for its effectiveness in image feature extraction.
However, the standard ResNet-18 architecture is modified to disable in-place operations for ReLU layers, ensuring non-in-place activation functions.


Feature Extraction: The backbone CNN processes input images and extracts relevant image features.
These features capture the visual information necessary for the classification task.


Transformer Encoder: After feature extraction, the model passes the feature maps through a Transformer encoder.
Transformers are originally designed for sequence data, but here, they are adapted for spatial data (feature maps).
The Transformer encoder is responsible for capturing relationships and dependencies between different regions of the image.


Classification Head: Following the Transformer encoder, the model uses a linear classifier to make predictions.
The encoder's output is averaged along the sequence dimension, resulting in a fixed-size representation of the image features.
This representation is then fed into the classifier, which outputs class probabilities.
"""


# Custom ResNet without in-place operations
from torchvision.models.resnet import BasicBlock, ResNet


class CustomResNet(ResNet):
    """
        Custom ResNet model that disables in-place operations for ReLU layers.
       
        Args:
            block (nn.Module): The residual block module.
            layers (list): List specifying the number of blocks in each layer.
            num_classes (int, optional): Number of output classes. Default is 1000.
            zero_init_residual (bool, optional): Whether to zero-initialize residual connections. Default is False.
            groups (int, optional): Number of groups for the 3x3 convolution layers. Default is 1.
            width_per_group (int, optional): Number of channels per group for the convolution layers. Default is 64.
            replace_stride_with_dilation (list or None, optional): Replace stride with dilation in the layers.
            norm_layer (nn.Module or None, optional): Normalization layer to use. Default is None.
        """
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
    """
    Custom ResNet-18 model that uses the CustomResNet class.
   
    Args:
        pretrained (bool, optional): Whether to load pre-trained weights. Default is False.
        progress (bool, optional): Whether to display a download progress bar. Default is True.
   
    Returns:
        CustomResNet: Custom ResNet-18 model.
    """
    return CustomResNet(BasicBlock, [2, 2, 2, 2], **kwargs)


# Hybrid CNN-Transformer Model
class CNNTransformer(nn.Module):
    """
        Custom CNN-Transformer model for a classification task.
       
        Args:
            num_classes (int, optional): Number of output classes. Default is 2.
            num_transformer_layers (int, optional): Number of transformer layers. Default is 3.
        """
    def __init__(self, num_classes=2, num_transformer_layers=3):
        super(CNNTransformer, self).__init__()
        self.backbone = resnet18(pretrained=True)
        # Remove the fully connected layer (since we'll add our own layers after)
        self.backbone = nn.Sequential(*list(self.backbone.children())[:-1])
       
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=512, nhead=8), num_layers=num_transformer_layers)
       
        self.classifier = nn.Linear(512, num_classes)


    def forward(self, x):
        x = self.backbone(x)
        x = torch.flatten(x, start_dim=2)  # Flatten to (B,C,N)
        x = x.permute(2, 0, 1)  # Permute to (N,B,C)
        x = self.transformer(x)
        x = x.mean(dim=0)  # Average along the sequence dimension
        x = self.classifier(x)
        return x


# Custom dataset loader to balance real and fake images
class CustomBalancedImageFolder(ImageFolder):
    def __init__(self, root, transform=None):
        super(CustomBalancedImageFolder, self).__init__(root, transform=transform)
        self.real_samples = [x for x in self.samples if x[1] == 0]
        self.fake_samples = [x for x in self.samples if x[1] == 1]
        min_samples = min(len(self.real_samples), len(self.fake_samples))
        self.samples = self.real_samples[:min_samples] + self.fake_samples[:min_samples]

def main():
    # Define your dataset paths
    data_dir = 'datasets/png_images'

    # Define image transforms
    transform = transforms.Compose([
        transforms.Resize((224, 224)),  # Resize images to the desired size
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    # Create datasets and data loaders for train, test, and validation
    # Create datasets and data loaders for train, test, and validation
    train_dataset = CustomBalancedImageFolder(os.path.join(data_dir, 'train'), transform=transform)
    test_dataset = CustomBalancedImageFolder(os.path.join(data_dir, 'test'), transform=transform)
    val_dataset = CustomBalancedImageFolder(os.path.join(data_dir, 'eval'), transform=transform)

    # DataLoader for validation and test
    trainloader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=2)
    valloader = DataLoader(val_dataset, batch_size=64, shuffle=True, num_workers=2)
    testloader = DataLoader(test_dataset, batch_size=64, shuffle=True, num_workers=2)

    # Define and train the model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CNNTransformer(num_classes=2).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    num_epochs = 10
    for epoch in tqdm(range(num_epochs)):  # Loop over the dataset multiple times
        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            inputs, labels = data[0].to(device), data[1].to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            if i % 200 == 199:  # Print every 200 mini-batches
                print('[%d, %5d] loss: %.3f' %
                    (epoch + 1, i + 1, running_loss / 200))
                running_loss = 0.0

    print('Finished Training')

    # Evaluation on the test set
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, labels in testloader:
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)

            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    print(f'Accuracy on the test set: {accuracy:.2f}%')




"""

# Custom dataset to balance real and fake images
class CustomBalancedDataset(Dataset):

    def __init__(self, root_dir, transform=None):
        self.real_dir = os.path.join(root_dir, 'real')
        self.fake_dir = os.path.join(root_dir, 'fake')
        self.transform = transform
       
        # Get lists of real and fake image file paths
        self.real_images = [os.path.join(self.real_dir, filename) for filename in os.listdir(self.real_dir)]
        self.fake_images = [os.path.join(self.fake_dir, filename) for filename in os.listdir(self.fake_dir)]
       
        # Ensure the same number of real and fake samples
        min_samples = min(len(self.real_images), len(self.fake_images))
        self.real_images = random.sample(self.real_images, min_samples)
        self.fake_images = random.sample(self.fake_images, min_samples)
       
        # Combine the datasets
        self.images = self.real_images + self.fake_images
        self.labels = [0] * min_samples + [1] * min_samples


    def __len__(self):
        return len(self.images)


    def __getitem__(self, idx):
        image_path = self.images[idx]
        image = Image.open(image_path).convert('RGB')
        label = self.labels[idx]


        if self.transform:
            image = self.transform(image)


        return image, label


# Create the balanced custom dataset
custom_dataset = CustomBalancedDataset(train_data_dir, transform=transform)


# Split the dataset into training and validation sets (if needed)
train_size = int(0.8 * len(custom_dataset))
val_size = len(custom_dataset) - train_size
train_dataset, test_dataset = random_split(custom_dataset, [train_size, val_size])


# Create data loaders for training and validation
batch_size = 64
trainloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
testloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True, num_workers=2)

"""
"""
# Create a directory to save the images on the remote server
save_dir = 'images'  # Replace with the directory where you want to save images on the server
os.makedirs(save_dir, exist_ok=True)


num_samples_to_save = 10  # Change this value to save more or fewer images
selected_indices = random.sample(range(len(train_dataset)), num_samples_to_save)


for i, idx in enumerate(selected_indices):
    sample, label = train_dataset[idx]
    image = sample.permute(1, 2, 0).numpy()  # Convert to (H, W, C) format as numpy array
    image = (image * 255).astype('uint8')  # Convert to 8-bit integer values
    image = Image.fromarray(image)
    image_filename = f"sample_{i}_label_{label}.png"
    image_path = os.path.join(save_dir, image_filename)
    image.save(image_path)


print(f"Saved {num_samples_to_save} images to {save_dir}.")
"""

"""
# Defining and training the model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = CNNTransformer(num_classes=2).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)


for epoch in tqdm(range(10)):  # Loop over the dataset multiple times
    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        inputs, labels = data[0].to(device), data[1].to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        if i % 200 == 199:  # Print every 200 mini-batches
            print('[%d, %5d] loss: %.3f' %
                  (epoch + 1, i + 1, running_loss / 200))
            running_loss = 0.0


print('Finished Training')


# Evaluate on the test set
correct = 0
total = 0
with torch.no_grad():
    for data in testloader:  # Use the testloader, not the test_dataset
        images, labels = data[0].to(device), data[1].to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()


print('Accuracy of the network on the test images: %.2f %%' % (100 * correct / total))

"""