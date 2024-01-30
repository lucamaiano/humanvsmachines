import os
import tqdm
import torch
import pandas as pd
from torchvision import transforms, datasets
import random
import numpy as np
from models.ResNet50 import ResNet50
from models.ResNet18 import ResNet18
from models.BasicCNN import SimpleCNN
from models.ViT import ViT
from models.Hybrid_PosEncoding import PositionalEncoding
from models.Hybrid_our_data import CNNTransformer
from scipy import stats
from pathlib import Path
import torch.nn.functional as F


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

    def __getitem__(self, index):
        path, target = self.samples[index]
        image = self.loader(path)
        if self.transform is not None:
            image = self.transform(image)

        return image, target, path


# Detect if we have a GPU available
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def evaluate(model, testloader):
    correct = 0
    total = 0
    fake_predictions = []
    real_predictions = []
    fake_confidences = []
    real_confidences = []
    image_predictions = {}

    with torch.no_grad():
        for inputs, labels, paths in tqdm.tqdm(testloader):
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs)

            # Count correct predictions
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            # Extract confidence values
            probabilities = F.softmax(outputs, dim=1)
            max_prob, predicted_class = torch.max(probabilities, dim=1)

            # Extend fake_predictions and real_predictions
            fake_predictions.extend(predicted[labels == 1].cpu().numpy())
            real_predictions.extend(predicted[labels == 0].cpu().numpy())

            # Costruisci il dizionario
            for path, pred_class, prob_percentage in zip(paths, predicted.cpu().numpy(), max_prob.cpu().numpy()):
                path_str = str(Path(path))  
                image_predictions[path_str] = [pred_class, prob_percentage]


    accuracy = (correct / total) * 100.0
    fake_accuracy = (fake_predictions.count(1) / (total // 2)) * 100.0  # Assuming equal number of 'fake' and 'real' samples
    real_accuracy = (real_predictions.count(0) / (total // 2)) * 100.0

    var_fake = np.var(fake_predictions)
    var_real = np.var(real_predictions)

    return accuracy, fake_accuracy, real_accuracy, var_fake, var_real, image_predictions



def create_dir(path):
    if not path.exists():
        #print(path)
        path.mkdir(parents=True, exist_ok=True)
        print(f'created folder: {path}')


#load the models
def load_and_test_models(model_names, dataset):
    loaded_models = {}
    
    for model_name in model_names:
        model_path = Path("weights/",model_name,'best.pth')
        print(model_path)
        if not model_path.exists():
            print(f"Model not found for {model_name}")
            continue
        
        # Load the model checkpoint
        #model = ResNet50()
        model = torch.load(model_path, map_location=device)#['state_dict']
        
        #model.load_state_dict(state_dict)
        model.to(device)
        loaded_models[model_name] = model
        
        # Test the model on the dataset
        accuracy, fake_accuracy, real_accuracy, var_fake, var_real, image_predictions = evaluate(model, dataset)
        print(f"Results for {model_name}:\n{accuracy}\n{(fake_accuracy, real_accuracy, var_fake, var_real)}")
        
        # Save results in the model's folder
        model_save_dir = "outputs"
        test_name_dir = "probabilities"
        if test_name_dir != "":
            model_folder = Path(model_save_dir,model_name, test_name_dir)
        else:
            model_folder = Path(model_save_dir,model_name)
        print(f"Output path: {model_folder}")
        create_dir(model_folder)
        results_file = Path(model_folder, f'{test_name_dir}.txt')
        with open(results_file, 'w') as f:
            f.write(f"Results for {model_name}:\n{accuracy}")
            f.write(f"fake_accuracy, real_accuracy, var_fake, var_real: {(fake_accuracy, real_accuracy, var_fake, var_real)}")
            f.write(f"Per image predictions:\n{image_predictions}")

        # Create a DataFrame containing the image_predictions
        df = pd.DataFrame(list(image_predictions.items()), columns=['File', 'Prediction_Confidence'])
        # Split the Prediction_Confidence in two separate columns: Prediction e Confidence
        df[['Prediction', 'Confidence']] = pd.DataFrame(df['Prediction_Confidence'].tolist(), index=df.index)
        df = df.drop(columns=['Prediction_Confidence'])
        # Save in an excel file
        excel_file_path = Path(model_folder, 'image_predictions.xlsx')  # Sostituisci con il percorso desiderato per il tuo file Excel
        df.to_excel(excel_file_path, index=False)

        
        
    return loaded_models

def main():
    # Parameters uncoment the data you want to test on 
    #data_dir = 'test_gender_women'  #Genders 
    data_dir = 'datasets/human_dataset/' #40 000 DATA(our generated dataset)
    #data_dir = 'CDDB/faces' # GAN generated data for generalization testing

    # hyperparameters
    input_size = 224
    workers = 1
    batch_size = 16

    # Create test dataset (always use the test dataset)
    test_dataset = CustomBalancedImageFolder(Path(data_dir, 'test'), transform=get_data_transform(input_size)['test'])

    # Create the test dataloader
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

