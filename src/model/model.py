import torch
import torch.nn as nn
import torchvision.models as models
from typing import Optional

def get_model(name: str = "mobilenet_v2", num_classes: int = 2, pretrained: bool = True):
    """
    Get model architecture for histopathology cancer detection
    
    Args:
        name: Model architecture name
        num_classes: Number of output classes
        pretrained: Whether to use pretrained weights (deprecated, using weights parameter)
    
    Returns:
        PyTorch model
    """
    weights = "DEFAULT" if pretrained else None
    
    if name == "mobilenet_v2":
        model = models.mobilenet_v2(weights=weights)
        # Replace the classifier
        model.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(model.last_channel, num_classes)
        )
    elif name == "resnet18":
        model = models.resnet18(weights=weights)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    elif name == "resnet50":
        model = models.resnet50(weights=weights)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    elif name == "efficientnet_b0":
        try:
            from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights
            model = efficientnet_b0(weights=EfficientNet_B0_Weights.DEFAULT if pretrained else None)
            model.classifier = nn.Sequential(
                nn.Dropout(0.2),
                nn.Linear(model.classifier[1].in_features, num_classes)
            )
        except ImportError:
            print("EfficientNet not available, falling back to MobileNetV2")
            model = models.mobilenet_v2(weights=weights)
            model.classifier = nn.Sequential(
                nn.Dropout(0.2),
                nn.Linear(model.last_channel, num_classes)
            )
    else:
        raise ValueError(f"Unknown model architecture: {name}")
    
    return model
