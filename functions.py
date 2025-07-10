
import torch
import torchvision.transforms as transforms
from PIL import Image
from models.eyenet import EyeNet
from models.swin import SwinTransformerWrapper

def load_model(name="EyeNet"):
    model = EyeNet() if name == "EyeNet" else SwinTransformerWrapper()
    model.load_state_dict(torch.load(f"models/{name.lower()}.pt", map_location="cpu"))
    model.eval()
    return model

def preprocess_image(img):
    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor()
    ])
    return transform(img).unsqueeze(0)

def predict_image(model, image_tensor):
    with torch.no_grad():
        output = model(image_tensor)
        prob = torch.softmax(output, dim=1)
        top_class = prob.argmax(dim=1).item()
        confidence = prob.max().item()
    return top_class, confidence
