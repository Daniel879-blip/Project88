
import os
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

def evaluate_model(model, dataloader, device):
    """
    Evaluate the model on a dataloader and return accuracy.
    """
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    return accuracy

def get_data_loaders(data_dir, image_size=224, batch_size=32):
    """
    Load train and test datasets from folders and return data loaders.
    Expects structure:
        data_dir/
            train/
                class1/
                class2/
            test/
                class1/
                class2/
    """
    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])
    ])

    train_path = os.path.join(data_dir, "train")
    test_path = os.path.join(data_dir, "test")

    train_dataset = datasets.ImageFolder(root=train_path, transform=transform)
    test_dataset = datasets.ImageFolder(root=test_path, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader
