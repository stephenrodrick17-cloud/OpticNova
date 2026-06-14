import torch
import torch.nn as nn
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights

class GlaucomaEfficientNet(nn.Module):
    """
    EfficientNet-B0 based classifier for Glaucoma detection.
    """
    def __init__(self, num_classes: int = 1, pretrained: bool = True):
        super(GlaucomaEfficientNet, self).__init__()
        
        # Load pre-trained EfficientNet-B0
        weights = EfficientNet_B0_Weights.DEFAULT if pretrained else None
        self.model = efficientnet_b0(weights=weights)
        
        # Replace the classifier head
        in_features = self.model.classifier[1].in_features
        self.model.classifier[1] = nn.Sequential(
            nn.Dropout(p=0.3, inplace=True),
            nn.Linear(in_features, num_classes)
        )
        
    def forward(self, x):
        return self.model(x)

if __name__ == "__main__":
    model = GlaucomaEfficientNet()
    # Dummy input
    x = torch.randn(2, 3, 512, 512)
    output = model(x)
    print(f"Output shape: {output.shape}")
