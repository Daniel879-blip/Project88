
import torch
import torchvision.transforms as transforms
from PIL import Image
import os

def load_dataset(split='val'):
    base_path = f"sample/{split}"
    images, labels = [], []
    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor()
    ])
    for label, folder in enumerate(os.listdir(base_path)):
        folder_path = os.path.join(base_path, folder)
        for file in os.listdir(folder_path):
            if file.endswith('.jpg') or file.endswith('.png'):
                img = Image.open(os.path.join(folder_path, file)).convert('RGB')
                images.append(transform(img))
                labels.append(label)
    return images, labels
