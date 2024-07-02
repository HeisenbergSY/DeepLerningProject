import torch
import torch.cuda.amp as amp
import matplotlib.pyplot as plt
import numpy as np
import random
import time
import logging
from torchvision.datasets import ImageFolder
from torchvision import transforms
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
import seaborn as sns
import platform
import psutil
from sklearn.model_selection import KFold

from data_loader import get_kfold_data_loaders, get_test_loader, HistogramEqualization
from model import VGG16Binary  # Import the VGG16Binary model
from train import train_model
from test import test_model
from inference import infer
from visualization import visualize_class_distribution
from torch.utils.data import DataLoader, Subset, SubsetRandomSampler

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

def main():
    # Start timing the execution
    start_time = time.time()

    # Set the random seed for reproducibility
    seed = 40
    set_seed(seed)

    train_dir = r'C:\Users\thepr\DeepLerningProject\chest_xray\train'
    test_dir = r'C:\Users\thepr\DeepLerningProject\chest_xray\test'

    # Set the best hyperparameters
    learning_rate = 0.001
    num_epochs = 20
    k_folds = 3

    # Define transforms
    transform_train = transforms.Compose([
        transforms.RandomRotation(30),
        transforms.RandomHorizontalFlip(),
        transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
        HistogramEqualization(),
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
        transforms.RandomVerticalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    transform_test_val = transforms.Compose([
        transforms.Resize((224, 224)),
        HistogramEqualization(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Get dataset and k-fold data loaders
    dataset = ImageFolder(root=train_dir, transform=transform_train)
    get_fold_data_loaders, k_folds = get_kfold_data_loaders(dataset, k_folds=k_folds, batch_size=80, num_workers=4)

    best_val_accuracy = 0
    best_model_state = None

    for fold in range(k_folds):
        print(f'Starting fold {fold + 1}/{k_folds}')
        train_loader, val_loader = get_fold_data_loaders(fold)

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f'Using device: {device}')

        # Initialize and train the model
        model = VGG16Binary().to(device)  # Use VGG16Binary model
        final_epoch, fold_val_accuracy = train_model(model, train_loader, val_loader, num_epochs=num_epochs, learning_rate=learning_rate, device=device)

        if fold_val_accuracy > best_val_accuracy:
            best_val_accuracy = fold_val_accuracy
            best_model_state = model.state_dict()

    # Save the best model
    if best_model_state:
        torch.save(best_model_state, 'best_model.pth')

    # Load the best model for testing
    best_model = VGG16Binary().to(device)  # Use VGG16Binary model
    best_model.load_state_dict(torch.load('best_model.pth'))

    # Test the model
    test_loader = get_test_loader(test_dir, batch_size=80, num_workers=4)
    accuracy, precision, recall, f1, auc_roc = test_model(best_model, test_loader, device=device)

    # Create confusion matrix
    y_true = []
    y_pred = []
    best_model.eval()
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.unsqueeze(1).float().to(device)
            outputs = best_model(images)
            predictions = outputs.round()
            y_true.extend(labels.cpu().numpy())
            y_pred.extend(predictions.cpu().numpy())

    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 7))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=['Normal', 'Pneumonia'], yticklabels=['Normal', 'Pneumonia'])
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    plt.show()

    # End timing the execution
    end_time = time.time()
    execution_time = end_time - start_time

    # Log hardware profile
    hw_profile = log_hardware_profile()

    # Write log file
    log_info = {
        "Learning Rate": learning_rate,
        "Model Used": "VGG16Binary",  # Updated model name
        "Execution Time (seconds)": execution_time,
        "Final Validation Accuracy": best_val_accuracy,
        "Test Accuracy": accuracy,
        "Test Precision": precision,
        "Test Recall": recall,
        "Test F1-score": f1,
        "Test AUC-ROC": auc_roc,
        "Hardware Profile": hw_profile
    }

    with open("training_log.txt", "w") as log_file:
        for key, value in log_info.items():
            if isinstance(value, dict):
                log_file.write(f"{key}:\n")
                for sub_key, sub_value in value.items():
                    log_file.write(f"  {sub_key}: {sub_value}\n")
            else:
                log_file.write(f"{key}: {value}\n")

if __name__ == '__main__':
    main()
