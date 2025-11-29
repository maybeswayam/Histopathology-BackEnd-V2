import torch
import torch.nn as nn
import torchvision.models as models

# Create a simple MobileNetV2 model
model = models.mobilenet_v2(weights=None)
model.classifier = nn.Sequential(
    nn.Dropout(0.2),
    nn.Linear(model.last_channel, 2)  # 2 classes: benign and malignant
)

# Initialize with random weights
model.eval()

# Create checkpoint
checkpoint = {
    'model_state_dict': model.state_dict(),
    'num_classes': 2,
    'input_size': 224,
    'classes': ['benign', 'malignant']
}

# Setup paths
import os
from pathlib import Path

# Get absolute path to models directory
current_dir = Path(__file__).parent.absolute()
models_dir = current_dir / 'models'
models_dir.mkdir(exist_ok=True)

# Save model
save_path = models_dir / 'model_best.pth'
torch.save(checkpoint, save_path)
print(f"Created placeholder model at: {save_path}")

# Verify file exists
if save_path.exists():
    print("✅ Model file created successfully")
else:
    print("❌ Failed to create model file")
print("Created placeholder model in models/model_best.pth")
