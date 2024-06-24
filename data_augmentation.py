import torch
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt

# Define the transformations
transform = transforms.Compose([
    transforms.RandomRotation(30),          # Randomly rotate images by 30 degrees
    transforms.RandomHorizontalFlip(),      # Randomly flip images horizontally
    transforms.RandomResizedCrop(224),      # Randomly crop and resize images to 224x224
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2), # Random color jitter
    transforms.ToTensor(),                  # Convert images to PyTorch tensors
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])   # Normalize images
])

# Load an example image
img = Image.open(r"C:\Users\thepr\Pictures\20240507_120518.jpg")

# Apply the transformations
img_transformed = transform(img)

# Convert tensor to image for visualization
img_transformed_show = transforms.ToPILImage()(img_transformed)

# Display the original and augmented images
fig, ax = plt.subplots(1, 2)
ax[0].imshow(img)
ax[0].set_title('Original Image')
ax[1].imshow(img_transformed_show)
ax[1].set_title('Augmented Image')
plt.show()
