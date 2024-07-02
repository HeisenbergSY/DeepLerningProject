import torch
from torch.utils.data import DataLoader, Subset, SubsetRandomSampler
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
import numpy as np
from collections import Counter
from PIL import ImageOps
from sklearn.model_selection import KFold  # Import KFold

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
    
    min_count = min(counter.values())
    
    indices = {class_idx: np.where(targets == class_idx)[0] for class_idx in counter.keys()}
    
    balanced_indices = []
    for class_idx, idx in indices.items():
        balanced_indices.extend(np.random.choice(idx, min_count, replace=False))
    
    balanced_dataset = Subset(dataset, balanced_indices)
    
    balanced_counter = Counter([targets[idx] for idx in balanced_indices])
    
    return balanced_dataset, counter, balanced_counter

def get_kfold_data_loaders(dataset, k_folds=3, batch_size=32, num_workers=4, augment=False):
    if augment:
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
    
    dataset.transform = transform_train

    kfold = KFold(n_splits=k_folds, shuffle=True)
    fold_indices = list(kfold.split(dataset))
    
    def get_fold_data_loaders(fold_index):
        train_idx, val_idx = fold_indices[fold_index]
        train_sampler = SubsetRandomSampler(train_idx)
        val_sampler = SubsetRandomSampler(val_idx)
        
        train_loader = DataLoader(dataset, batch_size=batch_size, sampler=train_sampler, num_workers=num_workers)
        val_loader = DataLoader(dataset, batch_size=batch_size, sampler=val_sampler, num_workers=num_workers)
        
        return train_loader, val_loader

    return get_fold_data_loaders, k_folds

def get_test_loader(test_dir, batch_size=32, num_workers=4):
    transform_test = transforms.Compose([
        transforms.Resize((224, 224)),
        HistogramEqualization(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    test_dataset = ImageFolder(root=test_dir, transform=transform_test)
    test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    return test_loader
