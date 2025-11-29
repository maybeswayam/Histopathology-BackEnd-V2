import torch
import os

model_best_path = "models/model_best.pth"
checkpoint_path = "../vansh-model/models/models/checkpoint.pth"

print("Comparing two model checkpoints:\n")

for name, path in [("model_best.pth", model_best_path), ("checkpoint.pth", checkpoint_path)]:
    if not os.path.exists(path):
        print(f"❌ {name} not found at {path}")
        continue
    
    print(f"\n{'='*50}")
    print(f"{name}")
    print('='*50)
    
    checkpoint = torch.load(path, map_location='cpu')
    
    if isinstance(checkpoint, dict):
        print(f"Keys: {list(checkpoint.keys())}")
        if 'epoch' in checkpoint:
            print(f"  Epoch: {checkpoint['epoch']}")
        if 'best_acc' in checkpoint:
            print(f"  Best Accuracy: {checkpoint['best_acc']:.4f}")
        if 'train_acc' in checkpoint:
            print(f"  Train Accuracy: {checkpoint['train_acc']:.4f}")
        if 'val_acc' in checkpoint:
            print(f"  Val Accuracy: {checkpoint['val_acc']:.4f}")
        
        # Check if state_dict exists and sample a weight value
        if 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
            first_key = list(state_dict.keys())[0]
            first_weight_sum = state_dict[first_key].sum().item()
            print(f"  First layer weight sum: {first_weight_sum:.6f}")
