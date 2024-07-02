import torch
from torch.utils.data import DataLoader, Subset
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
import numpy as np
from collections import Counter
from PIL import ImageOps

# Define the HistogramEqualization transformation
class HistogramEqualization:
    def __call__(self, img):
        return ImageOps.equalize(img)

def count_class_elements(dataset):
    targets = np.array(dataset.targets)
    counter = Counter(targets)
    return counter

def undersample_dataset(dataset):
    targets = np.array(dataset.targets)
    counter = Counter(targets)
    
    # Find the minority class count
    min_count = min(counter.values())
    
    # Collect indices for each class
    indices = {class_idx: np.where(targets == class_idx)[0] for class_idx in counter.keys()}
    
    # Randomly select indices to balance classes
    balanced_indices = []
    for class_idx, idx in indices.items():
        balanced_indices.extend(np.random.choice(idx, min_count, replace=False))
    
    # Create a subset of the dataset
    balanced_dataset = Subset(dataset, balanced_indices)
    
    balanced_counter = Counter([targets[idx] for idx in balanced_indices])
    
    return balanced_dataset, counter, balanced_counter

def get_data_loaders(train_dir, val_dir, test_dir, batch_size=32, augment=False, undersample=False):
    if augment:
        transform_train = transforms.Compose([
            transforms.RandomRotation(30),
            transforms.RandomHorizontalFlip(),
            transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),  # Reduced zoom effect
            HistogramEqualization(),
            transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
            transforms.RandomVerticalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    else:
        transform_train = transforms.Compose([
            transforms.Resize((224, 224)),
            HistogramEqualization(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    transform_test_val = transforms.Compose([
        transforms.Resize((224, 224)),
        HistogramEqualization(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    train_dataset = ImageFolder(root=train_dir, transform=transform_train)
    val_dataset = ImageFolder(root=val_dir, transform=transform_test_val)
    test_dataset = ImageFolder(root=test_dir, transform=transform_test_val)

    class_names = train_dataset.classes

    before_count = count_class_elements(train_dataset)
    after_count = before_count  # Default to before_count if undersampling is not done

    if undersample:
        balanced_train_dataset, before_count, after_count = undersample_dataset(train_dataset)
        train_loader = DataLoader(dataset=balanced_train_dataset, batch_size=batch_size, shuffle=True)
    else:
        train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
    
    val_loader = DataLoader(dataset=val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader, before_count, after_count, class_names
