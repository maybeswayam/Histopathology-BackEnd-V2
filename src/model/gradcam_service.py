import torch
import torch.nn.functional as F
import cv2
import numpy as np
from PIL import Image


def get_gradcam_standard(model, input_tensor, target_class=None):
    """
    Generate Grad-CAM visualization using the standard approach.
    This matches the implementation from generate_gradcam.py
    
    Args:
        model: PyTorch model (should be in eval mode)
        input_tensor: Input image tensor (1, C, H, W) - should be on same device as model
        target_class: Target class index (None = use predicted class)
    
    Returns:
        Normalized Grad-CAM heatmap as numpy array (224, 224)
    """
    # Ensure model is in eval mode and input is on correct device
    model.eval()
    device = next(model.parameters()).device
    input_tensor = input_tensor.to(device)
    input_tensor.requires_grad_(True)

    activations = None
    gradients = None

    def forward_hook(module, input, output):
        nonlocal activations
        activations = output.detach()

    def backward_hook(module, grad_input, grad_output):
        nonlocal gradients
        gradients = grad_output[0]

    # Use the same layer as generate_gradcam.py: model.features[-4]
    # For MobileNetV2, this targets a deeper convolutional layer
    if hasattr(model, 'features'):
        target_layer = model.features[-4]
    elif hasattr(model, 'layer4'):  # ResNet
        target_layer = model.layer4[-1]
    else:
        # Fallback: try to find the last convolutional layer
        for module in reversed(list(model.modules())):
            if isinstance(module, torch.nn.Conv2d):
                target_layer = module
                break
        else:
            raise ValueError("Could not find appropriate target layer for Grad-CAM")

    fwd_handle = target_layer.register_forward_hook(forward_hook)
    bwd_handle = target_layer.register_full_backward_hook(backward_hook)

    try:
        output = model(input_tensor)
        if target_class is None:
            target_class = output.argmax(dim=1).item()

        model.zero_grad()
        class_score = output[0, target_class]
        class_score.backward()

        fwd_handle.remove()
        bwd_handle.remove()

    except Exception as e:
        fwd_handle.remove()
        bwd_handle.remove()
        raise e

    if activations is None or gradients is None:
        raise ValueError(f"Failed to capture activations/gradients from {target_layer}. Try different layer.")

    # Compute Grad-CAM: weight activations by mean pooled gradients
    weights = torch.mean(gradients, dim=(2, 3), keepdim=True)
    cam = torch.sum(weights * activations, dim=1).squeeze()
    cam = F.relu(cam)

    # Interpolate to 224x224
    cam = F.interpolate(
        cam.unsqueeze(0).unsqueeze(0),
        size=(224, 224),
        mode='bilinear',
        align_corners=False
    ).squeeze().cpu().numpy()

    # Normalize
    if cam.max() > cam.min():
        cam = (cam - cam.min()) / (cam.max() - cam.min())
    else:
        cam = np.zeros_like(cam)

    return cam


class GradCAM:
    """Wrapper class for Grad-CAM computation"""
    
    def __init__(self, model, target_layer=None):
        """
        Initialize GradCAM
        
        Args:
            model: PyTorch model
            target_layer: Optional target layer (if None, will auto-detect)
        """
        self.model = model
        self.target_layer = target_layer

    def __call__(self, input_tensor, class_idx=None):
        """
        Generate Grad-CAM heatmap
        
        Args:
            input_tensor: Input image tensor
            class_idx: Target class index (None = use predicted class)
        
        Returns:
            Normalized heatmap as numpy array
        """
        return get_gradcam_standard(self.model, input_tensor, class_idx)


def overlay_heatmap(heatmap, original_image, alpha=0.4, colormap=cv2.COLORMAP_JET):
    """
    Overlays a heatmap onto the original image.
    Matches the overlay approach from generate_gradcam.py
    
    Args:
        heatmap: Grad-CAM heatmap (numpy array, 0-1 range)
        original_image: PIL Image
        alpha: Overlay transparency (0-1)
        colormap: OpenCV colormap
    
    Returns:
        PIL Image with overlay
    """
    # Resize image to match heatmap size (224x224) or resize heatmap to image
    img_np = np.array(original_image.resize((224, 224)))
    
    # Convert heatmap to color map
    heatmap_uint8 = np.uint8(255 * heatmap)
    heatmap_colored = cv2.applyColorMap(heatmap_uint8, colormap)
    heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)

    # Overlay: matches generate_gradcam.py approach
    overlay = (1 - alpha) * img_np.astype(np.float32) + alpha * heatmap_colored.astype(np.float32)
    overlay = np.clip(overlay, 0, 255).astype(np.uint8)

    return Image.fromarray(overlay)
