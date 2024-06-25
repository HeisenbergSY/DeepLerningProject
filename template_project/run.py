# run.py
import torch
from data_loader import get_data_loaders
from model import MobileNetV3Binary
from train import train_model
from test import test_model
from inference import infer

def main():
    train_dir = r'C:\Users\thepr\DeepLerningProject\chest_xray\train'
    val_dir = r'C:\Users\thepr\DeepLerningProject\chest_xray\val'
    test_dir = r'C:\Users\thepr\DeepLerningProject\chest_xray\test'

    # Enable data augmentation
    train_loader, val_loader, test_loader = get_data_loaders(train_dir, val_dir, test_dir, batch_size=32, augment=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = MobileNetV3Binary().to(device)

    train_model(model, train_loader, val_loader, num_epochs=10, learning_rate=0.001, device=device)
    test_model(model, test_loader, device=device)

    # Example inference
    # Assuming `image` is a preprocessed image tensor
    # image = ...
    # print(f'Prediction: {infer(model, image, device)}')

if __name__ == '__main__':
    main()
