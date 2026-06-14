import torch
import torch.nn.functional as F
import numpy as np

class GradCAM:
    """
    Grad-CAM implementation for PyTorch models.
    """
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None

        # Hook to extract activations
        self.target_layer.register_forward_hook(self.save_activation)
        # Hook to extract gradients
        self.target_layer.register_full_backward_hook(self.save_gradient)

    def save_activation(self, module, input, output):
        self.activations = output

    def save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0]

    def generate_heatmap(self, input_tensor, class_idx=None):
        """
        Generates the Grad-CAM heatmap.
        """
        self.model.eval()
        
        # Forward pass
        output = self.model(input_tensor)
        
        if class_idx is None:
            # If no class provided, use the class with highest probability
            class_idx = output.argmax(dim=1).item()
            
        # Backward pass
        self.model.zero_grad()
        target = output[0, class_idx]
        target.backward()
        
        # Calculate weights from gradients
        pooled_gradients = torch.mean(self.gradients, dim=[0, 2, 3])
        
        # Apply weights to activations
        for i in range(self.activations.shape[1]):
            self.activations[:, i, :, :] *= pooled_gradients[i]
            
        # Create heatmap
        heatmap = torch.mean(self.activations, dim=1).squeeze()
        heatmap = F.relu(heatmap)
        
        # Normalize heatmap
        heatmap /= torch.max(heatmap) + 1e-8
        
        return heatmap.cpu().detach().numpy()
