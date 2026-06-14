import torch
import torch.nn as nn
from torchvision.models import resnet50, ResNet50_Weights

class GlaucomaResNet(nn.Module):
    """
    ResNet50 based classifier for Glaucoma detection.
    """
    def __init__(self, num_classes: int = 1, pretrained: bool = True):
        super(GlaucomaResNet, self).__init__()
        
        # Load pre-trained ResNet50
        weights = ResNet50_Weights.DEFAULT if pretrained else None
        self.model = resnet50(weights=weights)
        
        # Replace the fully connected layer
        in_features = self.model.fc.in_features
        self.model.fc = nn.Sequential(
            nn.Dropout(p=0.3, inplace=True),
            nn.Linear(in_features, num_classes)
        )
        
    def forward(self, x):
        return self.model(x)

if __name__ == "__main__":
    model = GlaucomaResNet()
    # Dummy input
    x = torch.randn(2, 3, 512, 512)
    output = model(x)
    print(f"Output shape: {output.shape}")
