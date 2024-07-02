# inference.py
import torch

def infer(model, image, device='cuda'):
    model.eval()
    with torch.no_grad():
        image = image.to(device)
        output = model(image)
        prediction = output.round()
    return prediction.item()
