import torch
from data_loader import get_data_loaders
from model import MobileNetV3Binary
from train import train_model
from test import test_model
from inference import infer


def main():
    train_dir = 'path/to/train'
    val_dir = 'path/to/val'
    test_dir = 'path/to/test'

    train_loader, val_loader, test_loader = get_data_loaders(train_dir, val_dir, test_dir)

    model = MobileNetV3Binary()
    model = model.to(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))

    train_model(model, train_loader, val_loader, num_epochs=10, learning_rate=0.001)
    test_model(model, test_loader)

    # Example inference
    # Assuming `image` is a preprocessed image tensor
    # image = ...
    # print(f'Prediction: {infer(model, image)}')


if __name__ == '__main__':
    main()