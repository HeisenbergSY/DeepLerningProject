import torch
import matplotlib.pyplot as plt
import numpy as np
import random
import time
import logging
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import platform
import psutil
from data_loader import get_data_loaders
from model import MobileNetV3Binary
from train import train_model
from test import test_model
from inference import infer
from visualization import visualize_class_distribution

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

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
    val_dir = r'C:\Users\thepr\DeepLerningProject\chest_xray\val'
    test_dir = r'C:\Users\thepr\DeepLerningProject\chest_xray\test'

    # Set the best hyperparameters
    learning_rate = 0.001
    num_epochs = 20

    # Data augmentations
    data_augmentations = [
        "RandomRotation(30)",
        "RandomHorizontalFlip()",
        "RandomResizedCrop(224, scale=(0.8, 1.0))",  # Reduced zoom effect
        "HistogramEqualization()",
        "RandomAffine(degrees=0, translate=(0.1, 0.1))",
        "RandomVerticalFlip()"
    ]

    # Get data loaders
    train_loader, val_loader, test_loader, before_count, after_count, class_names = get_data_loaders(
        train_dir, val_dir, test_dir, batch_size=32, augment=True, undersample=True
    )

    # Visualize class distribution before and after downsampling
    visualize_class_distribution(before_count, after_count, class_names)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')

    # Initialize and train the model
    model = MobileNetV3Binary().to(device)
    final_epoch, best_val_accuracy = train_model(model, train_loader, val_loader, num_epochs=num_epochs, learning_rate=learning_rate, device=device)
    accuracy, precision, recall, f1, auc_roc = test_model(model, test_loader, device=device)
    print(f'Test Accuracy: {accuracy}%')
    print(f'Test Precision: {precision}%')
    print(f'Test Recall: {recall}%')
    print(f'Test F1-score: {f1}%')
    print(f'Test AUC-ROC: {auc_roc}%')

    # Save the model
    torch.save(model.state_dict(), 'best_model.pth')

    # End timing the execution
    end_time = time.time()
    execution_time = end_time - start_time

    # Log hardware profile
    hw_profile = log_hardware_profile()

    # Write log file
    log_info = {
        "Learning Rate": learning_rate,
        "Data Augmentations": data_augmentations,
        "Model Used": "MobileNetV3Binary",
        "Execution Time (seconds)": execution_time,
        "Number of Epochs": final_epoch,
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
