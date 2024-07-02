import os
import yaml
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
from torchvision.models import vgg16, VGG16_Weights, resnet50, ResNet50_Weights
from sklearn.model_selection import KFold, GridSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from torch.utils.data import DataLoader, Dataset, Subset
from PIL import Image, ImageOps
import numpy as np
from collections import Counter
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
import copy
import random
import time
import platform
import psutil

# Set random seed for reproducibility
def set_seed(seed=42):
    torch.manual_seed(seed)
    np.random.seed(seed)

set_seed()

# Define PneumoniaDataset class with advanced preprocessing
class PneumoniaDataset(Dataset):
    def __init__(self, root_dir, transform=None, augment=False):  # Use double underscores
        self.root_dir = root_dir
        self.transform = transform
        self.augment = augment
        self.samples = []
        for label in ['NORMAL', 'PNEUMONIA']:
            label_dir = os.path.join(root_dir, label)
            for file in os.listdir(label_dir):
                self.samples.append((os.path.join(label_dir, file), 0 if label == 'NORMAL' else 1))

    def __len__(self):  # Use double underscores
        return len(self.samples)

    def __getitem__(self, idx):  # Use double underscores
        img_path, label = self.samples[idx]
        image = Image.open(img_path).convert('RGB')
        image = ImageOps.equalize(image)  # Histogram Equalization
        if self.transform:
            image = self.transform(image)
        if self.augment:
            image = self.apply_augmentation(image)
        return image, label

    def apply_augmentation(self, image):
        augment_transforms = transforms.Compose([
            transforms.RandomRotation(20),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2),
        ])
        return augment_transforms(image)


# Data augmentation and normalization
data_transforms = {
    'train': transforms.Compose([
        transforms.RandomRotation(20),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    'val': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'test': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}

# Load datasets
data_dir = "C:/Users/thepr/DeepLerningProject/chest_xray"
train_dataset = PneumoniaDataset(root_dir=os.path.join(data_dir, 'train'), transform=data_transforms['train'], augment=True)
val_dataset = PneumoniaDataset(root_dir=os.path.join(data_dir, 'val'), transform=data_transforms['val'])
test_dataset = PneumoniaDataset(root_dir=os.path.join(data_dir, 'test'), transform=data_transforms['test'])

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# plot class distribution

def plot_class_distribution(dataset, title='Class Distribution'):
    labels = [label for _, label in dataset.samples]
    counter = Counter(labels)
    plt.figure(figsize=(6, 4))
    plt.bar(counter.keys(), counter.values(), color=['blue', 'orange'])
    plt.xticks([0, 1], ['Normal', 'Pneumonia'])
    plt.xlabel('Class')
    plt.ylabel('Frequency')
    plt.title(title)
    plt.show()

# Visualize class distribution before resampling
plot_class_distribution(train_dataset, title='Class Distribution Before Resampling')
# Resample the dataset
def resample_dataset(dataset, batch_size=100):
    labels = [label for _, label in dataset.samples]
    counter = Counter(labels)
    print("Original class distribution:", counter)

    if counter[0] > counter[1]:
        majority_class = 0
        minority_class = 1
    else:
        majority_class = 1
        minority_class = 0

    oversample = SMOTE(sampling_strategy='minority')
    undersample = RandomUnderSampler(sampling_strategy='majority')

    resampled_samples = []
    temp_dir = '/tmp/resampled_images'
    os.makedirs(temp_dir, exist_ok=True)

    for i in range(0, len(dataset.samples), batch_size):
        batch_samples = dataset.samples[i:i + batch_size]
        image_paths = [img_path for img_path, _ in batch_samples]
        labels = [label for _, label in batch_samples]
        print(f"Batch {i}: Class distribution - {Counter(labels)}")

        if len(set(labels)) < 2:
            print(f"Skipping batch {i} because it contains only one class")
            continue

        images = []
        for img_path in image_paths:
            try:
                if os.path.exists(img_path):
                    img = Image.open(img_path).convert('RGB').resize((128, 128))
                    images.append(np.array(img))
                else:
                    print(f"Image not found: {img_path}")
            except Exception as e:
                print(f"Error loading image {img_path}: {e}")
                continue

        if not images:
            continue

        X = np.array(images)
        y = np.array(labels)

        try:
            X_res, y_res = oversample.fit_resample(X.reshape(X.shape[0], -1), y)
            X_res, y_res = undersample.fit_resample(X_res, y_res)
            print(f"Batch {i}: Resampled class distribution - {Counter(y_res)}")
        except ValueError as ve:
            print(f"Resampling error in batch {i}: {ve}")
            continue

        for x, y in zip(X_res, y_res):
            x = x.reshape(128, 128, 3)
            img = Image.fromarray(x.astype('uint8'), 'RGB')
            temp_img_path = os.path.join(temp_dir, f'{len(resampled_samples)}.png')
            img.save(temp_img_path)
            resampled_samples.append((temp_img_path, y))

    print("Number of resampled samples:", len(resampled_samples))
    return resampled_samples

resampled_samples = resample_dataset(train_dataset)
if resampled_samples:
    train_dataset.samples = resampled_samples
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
else:
    print("Warning: No samples were resampled. Check for errors in the resampling process.")

plot_class_distribution(train_dataset, title='Class Distribution After Resampling')
# Model training and evaluation functions
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def train_model(model, criterion, optimizer, dataloaders, num_epochs=25):
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        print(f'Epoch {epoch}/{num_epochs - 1}')
        print('-' * 10)

        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            running_corrects = 0

            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device).long()  # Ensure labels are of type torch.LongTensor

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            epoch_acc = running_corrects.double() / len(dataloaders[phase].dataset)

            print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

    model.load_state_dict(best_model_wts)
    return model

def evaluate_model(model, dataloader):
    model.eval()
    all_labels = []
    all_preds = []

    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)

            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())

    accuracy = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds)
    recall = recall_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds)
    auc_roc = roc_auc_score(all_labels, all_preds)

    return accuracy, precision, recall, f1, auc_roc

# Fine-tuning VGG-16
vgg_model = vgg16(weights=VGG16_Weights.DEFAULT)
for param in vgg_model.features.parameters():
    param.requires_grad = False
for param in vgg_model.classifier[:-1].parameters():
    param.requires_grad = True
vgg_model.classifier[6] = nn.Linear(vgg_model.classifier[6].in_features, 2)
vgg_model = vgg_model.to(device)

# Fine-tuning more layers in ResNet-50
resnet_model = resnet50(weights=ResNet50_Weights.DEFAULT)
for param in resnet_model.parameters():
    param.requires_grad = True
resnet_model.fc = nn.Linear(resnet_model.fc.in_features, 2)
resnet_model = resnet_model.to(device)

dataloaders = {
    'train': train_loader,
    'val': val_loader,
    'test': test_loader
}
criterion = nn.CrossEntropyLoss()

# Define optimizers
vgg_optimizer = torch.optim.SGD(vgg_model.parameters(), lr=1e-3, momentum=0.9)
resnet_optimizer = torch.optim.AdamW(resnet_model.parameters(), lr=1e-4)

# Train models
vgg_model = train_model(vgg_model, criterion, vgg_optimizer, dataloaders, num_epochs=5)
resnet_model = train_model(resnet_model, criterion, resnet_optimizer, dataloaders, num_epochs=5)

# Evaluate models
vgg_metrics = evaluate_model(vgg_model, test_loader)
resnet_metrics = evaluate_model(resnet_model, test_loader)

print(f'VGG-16 Metrics: {vgg_metrics}')
print(f'ResNet-50 Metrics: {resnet_metrics}')
# Perform k-fold cross-validation
def k_fold_cross_validation(model, dataset, criterion, optimizer, k=10, num_epochs=20):
    kf = KFold(n_splits=k, shuffle=True, random_state=42)
    metrics = []

    for train_index, val_index in kf.split(dataset):
        train_subset = Subset(dataset, train_index)
        val_subset = Subset(dataset, val_index)

        train_loader = DataLoader(train_subset, batch_size=32, shuffle=True)
        val_loader = DataLoader(val_subset, batch_size=32, shuffle=False)

        dataloaders = {'train': train_loader, 'val': val_loader}

        model = train_model(model, criterion, optimizer, dataloaders, num_epochs)
        fold_metrics = evaluate_model(model, val_loader)
        metrics.append(fold_metrics)

    avg_metrics = np.mean(metrics, axis=0)
    return avg_metrics

# Example usage with VGG-16
vgg_avg_metrics = k_fold_cross_validation(vgg_model, train_dataset, criterion, vgg_optimizer, k=10, num_epochs=20)
print(f'VGG-16 K-Fold Cross-Validation Metrics: {vgg_avg_metrics}')

# Perform bootstrap aggregating (bagging)
def bootstrap_aggregating(models, optimizers, dataset, n_estimators=5, num_epochs=15):
    all_preds = []
    all_labels = []

    for _ in range(n_estimators):
        bootstrap_sample = Subset(dataset, np.random.choice(len(dataset), len(dataset), replace=True))
        loader = DataLoader(bootstrap_sample, batch_size=32, shuffle=True)

        for model, optimizer in zip(models, optimizers):
            model = train_model(model, criterion, optimizer, {'train': loader, 'val': val_loader}, num_epochs)
            accuracy, precision, recall, f1, auc_roc = evaluate_model(model, test_loader)
            all_preds.append(accuracy)
            all_labels.append([label for _, label in test_loader.dataset.samples])

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    final_preds = np.apply_along_axis(lambda x: np.bincount(x).argmax(), axis=0, arr=all_preds)

    accuracy = accuracy_score(all_labels[0], final_preds)
    precision = precision_score(all_labels[0], final_preds)
    recall = recall_score(all_labels[0], final_preds)
    f1 = f1_score(all_labels[0], final_preds)
    auc_roc = roc_auc_score(all_labels[0], final_preds)

    return accuracy, precision, recall, f1, auc_roc
print("Detailed Analysis:")
print("1. Accuracy: ")
print(f"   VGG-16: {vgg_acc:.4f}, ResNet-50: {resnet_acc:.4f}")
if resnet_acc > vgg_acc:
    print("   ResNet-50 performed better in terms of accuracy.")
else:
    print("   VGG-16 performed better in terms of accuracy.")

print("\n2. Precision: ")
print(f"   VGG-16: {vgg_prec:.4f}, ResNet-50: {resnet_prec:.4f}")
if resnet_prec > vgg_prec:
    print("   ResNet-50 has higher precision, meaning it has fewer false positives.")
else:
    print("   VGG-16 has higher precision, meaning it has fewer false positives.")

print("\n3. Recall: ")
print(f"   VGG-16: {vgg_recall:.4f}, ResNet-50: {resnet_recall:.4f}")
if resnet_recall > vgg_recall:
    print("   ResNet-50 has higher recall, meaning it has fewer false negatives.")
else:
    print("   VGG-16 has higher recall, meaning it has fewer false negatives.")

print("\n4. F1 Score: ")
print(f"   VGG-16: {vgg_f1:.4f}, ResNet-50: {resnet_f1:.4f}")
if resnet_f1 > vgg_f1:
    print("   ResNet-50 has a better balance between precision and recall.")
else:
    print("   VGG-16 has a better balance between precision and recall.")

print("\n5. AUC-ROC: ")
print(f"   VGG-16: {vgg_auc:.4f}, ResNet-50: {resnet_auc:.4f}")
if resnet_auc > vgg_auc:
    print("   ResNet-50 has a better overall classification performance.")
else:
    print("   VGG-16 has a better overall classification performance.")
# Log model performance
def log_model_performance(model_name, metrics, model):
    accuracy, precision, recall, f1, auc_roc = metrics
    print(f'{model_name} Metrics:')
    print(f'Accuracy: {accuracy:.4f}')
    print(f'Precision: {precision:.4f}')
    print(f'Recall: {recall:.4f}')
    print(f'F1 Score: {f1:.4f}')
    print(f'AUC-ROC: {auc_roc:.4f}')

    # Log parameters and hardware profile
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'{model_name} Number of trainable parameters: {num_params}')
    print(f'Running on device: {device}')

# Log model parameters and hardware profile
def log_model_params_and_device(model, model_name):
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'\n{model_name} Parameters and Hardware Profile:')
    print(f'Number of trainable parameters: {num_params}')
    print(f'Running on device: {device}')

# Save the best model
def save_model(model, path='best_model.pth'):
    torch.save(model.state_dict(), path)

# Inference function
def inference(model, image_dir, transform):
    model.eval()
    image_files = []
    for root, _, files in os.walk(image_dir):
        for file in files:
            if file.endswith(('.jpg', '.jpeg', '.png')):
                image_files.append(os.path.join(root, file))

    if not image_files:
        print("No image files found in the directory.")
        return None, None

    random_image_file = random.choice(image_files)
    image = Image.open(random_image_file).convert('RGB')
    image = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(image)
        _, preds = torch.max(output, 1)

    return preds.item(), random_image_file

# Example inference
image_dir = '/content/drive/MyDrive/chest_xray/chest_xray/test'
prediction, image_name = inference(vgg_model, image_dir, data_transforms['test'])

if prediction is not None:
    print(f'Prediction for {image_name}: {"Pneumonia" if prediction == 1 else "Normal"}')