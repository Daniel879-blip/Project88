
from torchvision.models import swin_t
import torch.nn as nn

class SwinTransformerWrapper(nn.Module):
    def __init__(self, num_classes=3):
        super(SwinTransformerWrapper, self).__init__()
        self.model = swin_t(weights='DEFAULT')
        self.model.head = nn.Linear(self.model.head.in_features, num_classes)

    def forward(self, x):
        return self.model(x)
