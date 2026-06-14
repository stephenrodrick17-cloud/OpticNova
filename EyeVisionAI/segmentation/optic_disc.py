import torch
from .unet import UNet

class OpticDiscSegmenter:
    """
    Wrapper for Optic Disc Segmentation using U-Net.
    """
    def __init__(self, model_path: str = None, device: str = 'cuda' if torch.cuda.is_available() else 'cpu'):
        self.device = device
        self.model = UNet(n_channels=3, n_classes=1).to(self.device)
        
        if model_path:
            self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        
        self.model.eval()

    def predict(self, image_tensor: torch.Tensor) -> torch.Tensor:
        """
        Predicts the optic disc mask.
        
        Args:
            image_tensor (torch.Tensor): Preprocessed image tensor (B, C, H, W).
            
        Returns:
            torch.Tensor: Predicted binary mask.
        """
        with torch.no_grad():
            image_tensor = image_tensor.to(self.device)
            logits = self.model(image_tensor)
            probs = torch.sigmoid(logits)
            preds = (probs > 0.5).float()
        return preds
