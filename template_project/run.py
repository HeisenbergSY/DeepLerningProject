import torch
import torch.cuda.amp as amp
import matplotlib.pyplot as plt
import numpy as np
import random
import time
import logging
from torchvision.datasets import ImageFolder
from torchvision import transforms
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import platform
import psutil
from data_loader import get_data_loaders
from model import MobileNetV3Binary
from train import train_model
from test import test_model
from inference import infer
from visualization import visualize_class_distribution
from sklearn.model_selection import KFold
from torch.utils.data import DataLoader, Subset, SubsetRandomSampler
import itertools

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True  # Enable cuDNN benchmarking for improved performance

def show_augmented_images(data_loader):
    images, _ = next(iter(data_loader))
    images = images.numpy()
    fig, axes = plt.subplots(1, 5, figsize=(15, 3))
    for i in range(5):
        image = images[i].transpose((1, 2, 0))
        image = (image * 0.229 + 0.485).clip(0, 1)
        axes[i].imshow(image)
        axes[i].axis('off')
    plt.show()

def log_hardware_profile():
    hw_profile = {
        "Platform": platform.platform(),
        "Processor": platform.processor(),
        "RAM": f"{psutil.virtual_memory().total / (1024 ** 3):.2f} GB",
        "CUDA Available": torch.cuda.is_available(),
        "CUDA Device Count": torch.cuda.device_count(),
        "CUDA Device Name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "N/A"
    }
    return hw_profile

def get_kfold_data_loaders(dataset, k=3, batch_size=256):  # Increased batch size
    kf = KFold(n_splits=k, shuffle=True, random_state=42)
    fold_data_loaders = []

    for train_index, val_index in kf.split(dataset):
        train_subset = Subset(dataset, train_index)
        val_subset = Subset(dataset, val_index)
        
        train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=True)  # Increased num_workers
        val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False, num_workers=8, pin_memory=True)  # Increased num_workers
        
        fold_data_loaders.append((train_loader, val_loader))

    return fold_data_loaders

def main():
    # Start timing the execution
    start_time = time.time()

    # Set the random seed for reproducibility
    seed = 40
    set_seed(seed)

    train_dir = r'C:\Users\thepr\DeepLerningProject\chest_xray\train'
    test_dir = r'C:\Users\thepr\DeepLerningProject\chest_xray\test'

    # Set the hyperparameters
    learning_rate = 0.001
    weight_decay = 0.0001
    num_epochs = 1
    k = 3  # Number of folds for K-fold cross-validation

    # Data augmentations
    data_augmentations = [
        "RandomRotation(30)",
        "RandomHorizontalFlip()",
        "RandomResizedCrop(224, scale=(0.8, 1.0))",  # Reduced zoom effect
        "HistogramEqualization()",
        "RandomAffine(degrees=0, translate=(0.1, 0.1))",
        "RandomVerticalFlip()"
    ]

    # Get the dataset
    full_dataset = ImageFolder(root=train_dir, transform=transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ]))

    # Get test data loader
    test_dataset = ImageFolder(root=test_dir, transform=transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ]))
    test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False, num_workers=8, pin_memory=True)  # Increased batch size and num_workers

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')

    # K-fold Cross-Validation
    fold_data_loaders = get_kfold_data_loaders(full_dataset, k=k, batch_size=256)  # Increased batch size

    best_accuracy = 0
    best_params = {}
    hyperparam_tuning_results = []

    for fold, (train_loader, val_loader) in enumerate(fold_data_loaders):
        print(f"Training fold {fold + 1}/{k}")
        model = MobileNetV3Binary().to(device)
        final_epoch, val_accuracy = train_model(model, train_loader, val_loader, num_epochs=num_epochs, learning_rate=learning_rate, weight_decay=weight_decay, device=device)

        # Aggregate predictions
        accuracy, precision, recall, f1, auc_roc = test_model(model, test_loader, device=device)

        fold_results = {
            "fold": fold + 1,
            "test_accuracy": accuracy,
            "test_precision": precision,
            "test_recall": recall,
            "test_f1_score": f1,
            "test_auc_roc": auc_roc
        }
        hyperparam_tuning_results.append(fold_results)

        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_params = {
                "fold": fold + 1,
                "learning_rate": learning_rate,
                "num_epochs": num_epochs,
                "weight_decay": weight_decay
            }

        # Save the best model
        torch.save(model.state_dict(), f'best_model_fold_{fold + 1}.pth')

    print(f"Best parameters: {best_params}")
    print(f"Best accuracy: {best_accuracy}")

    # End timing the execution
    end_time = time.time()
    execution_time = end_time - start_time

    # Log hardware profile
    hw_profile = log_hardware_profile()

    # Write log file
    log_info = {
        "Best Parameters": best_params,
        "Best Accuracy": best_accuracy,
        "Data Augmentations": data_augmentations,
        "Model Used": "MobileNetV3Binary",
        "Execution Time (seconds)": execution_time,
        "Number of Epochs": num_epochs,
        "K-Fold Cross-Validation Results": hyperparam_tuning_results,
        "Hardware Profile": hw_profile
    }

    with open("training_log.txt", "w") as log_file:
        for key, value in log_info.items():
            if isinstance(value, list):
                log_file.write(f"{key}:\n")
                for item in value:
                    log_file.write(f"  {item}\n")
            elif isinstance(value, dict):
                log_file.write(f"{key}:\n")
                for sub_key, sub_value in value.items():
                    log_file.write(f"  {sub_key}: {sub_value}\n")
            else:
                log_file.write(f"{key}: {value}\n")

if __name__ == '__main__':
    main()
