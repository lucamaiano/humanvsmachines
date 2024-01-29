import os
import torch
from torchvision import transforms, datasets
import random
import numpy as np
from ResNet50 import ResNet50
from ResNet18 import ResNet18
from BasicCNN import SimpleCNN
from ViT import ViT
from Hybrid_PosEncoding import PositionalEncoding,CNNTransformer
from scipy import stats




# Set random seeds for reproducibility
random.seed(0)
np.random.seed(0)
torch.manual_seed(0)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False




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



# Detect if we have a GPU available
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")



def evaluate(model, testloader):
    correct = 0
    total = 0
    fake_predictions = []
    real_predictions = []

    with torch.no_grad():
        for inputs, labels in testloader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            fake_predictions.extend(predicted[labels == 1].cpu().numpy())
            real_predictions.extend(predicted[labels == 0].cpu().numpy())

    accuracy = (correct / total) * 100.0
    fake_accuracy = (fake_predictions.count(1) / (total // 2)) * 100.0  # Assuming equal number of 'fake' and 'real' samples
    real_accuracy = (real_predictions.count(0) / (total // 2)) * 100.0

    var_fake = np.var(fake_predictions)
    var_real = np.var(real_predictions)

    return accuracy, fake_accuracy, real_accuracy, var_fake, var_real



def create_dir(path):
    if not os.path.exists(path):
        #print(path)
        os.mkdir(path)
        print(f'created folder: {path}')
    else:
        print(f'directory {path} already exists')


#load the models
def load_and_test_models(model_names, dataset):
    loaded_models = {}
    
    for model_name in model_names:
        model_path = os.path.join("/RealFaces_w_StableDiffusion/",model_name,'best.pth')
        print(model_path)
        if not os.path.exists(model_path):
            print(f"Model not found for {model_name}")
            continue
        
        # Load the model checkpoint
        #model = ResNet50()
        model = torch.load(model_path, map_location=device)#['state_dict']
        
        #model.load_state_dict(state_dict)
        model.to(device)
        loaded_models[model_name] = model
        
        # Test the model on the dataset
        results = evaluate(model, dataset)
        print(f"Results for {model_name}: {results}")
        
        # Save results in the model's folder
        model_save_dir = "/RealFaces_w_StableDiffusion/"
        model_folder = os.path.join(model_save_dir,model_name, 'results_genders_w')
        print(model_folder)
        create_dir(model_folder)
        results_file = os.path.join(model_folder, 'results_genders_w.txt')
        with open(results_file, 'w') as f:
            f.write(f"Results for {model_name}: {results}")
        
    return loaded_models

def main():
    # Parameters
    #data_dir = '/RealFaces_w_StableDiffusion/test_gender_women'  #Genders 
    data_dir = '/RealFaces_w_StableDiffusion/datasets/png_images/' #40 000 DATA
    #data_dir = '/RealFaces_w_StableDiffusion/CDDB/faces' #Gan data for generalization test

    input_size = 224
    workers = 6
    batch_size = 32

    # Create test dataset
    test_dataset = CustomBalancedImageFolder(os.path.join(data_dir, 'test'), transform=get_data_transform(input_size)['test'])

    # Create training and validation dataloaders
    dataloaders_dict = {
        'test': torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=True, num_workers=workers)
    }

    # Define the model_names you want to test
    #model_names = ['ResNet50', 'ResNet18', 'custom_cnn_transformer_40','custom_cnn_transformer_PosEncofing','ViT', 'hybrid_PosEncoding_scratch']
    model_names = ['custom_cnn_transformer_40']


    # Load and test the models, and store the loaded_models dictionary
    loaded_models = load_and_test_models(model_names, dataloaders_dict['test'])

    for model_name, model in loaded_models.items():
        print(f"Loaded {model_name} model.")

if __name__ == '__main__':
    main()

