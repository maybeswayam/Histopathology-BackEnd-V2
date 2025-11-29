import torch

model_path = "models/model_best.pth"
print(f"Inspecting: {model_path}\n")

checkpoint = torch.load(model_path, map_location='cpu')

print(f"Type of checkpoint: {type(checkpoint)}")
print(f"\nIs it a dict? {isinstance(checkpoint, dict)}")

if isinstance(checkpoint, dict):
    print(f"\nKeys in checkpoint: {list(checkpoint.keys())}")
    for key in checkpoint.keys():
        print(f"  - {key}: {type(checkpoint[key])}")
else:
    print("\nCheckpoint is not a dict - it's likely the state_dict itself")
    print(f"Number of items: {len(checkpoint) if hasattr(checkpoint, '__len__') else 'N/A'}")

print("\n" + "="*50)
print("DIAGNOSIS:")
print("="*50)

if isinstance(checkpoint, dict):
    if 'model_state_dict' in checkpoint:
        print("✅ Found 'model_state_dict' key - this is the standard format")
    elif 'state_dict' in checkpoint:
        print("✅ Found 'state_dict' key")
    else:
        print("⚠️ Checkpoint is a dict but has no 'state_dict' or 'model_state_dict' key")
        print("   The dict might BE the state_dict")
        # Check if it looks like a state_dict (has layer names)
        keys = list(checkpoint.keys())
        if keys and any('weight' in k or 'bias' in k for k in keys[:5]):
            print("   ✅ Keys look like layer parameters - treating as state_dict")
else:
    print("⚠️ Checkpoint is not a dict - unusual format")
