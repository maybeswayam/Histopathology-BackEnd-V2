import torch
import os
from src.model.model_loader import get_model

def verify_model():
    print("=" * 50)
    print("MODEL VERIFICATION")
    print("=" * 50)
    
    model_path = "models/model_best.pth"
    if os.path.exists(model_path):
        print(f"✅ Model file found: {model_path}")
        file_size = os.path.getsize(model_path) / (1024 * 1024)
        print(f"   Size: {file_size:.2f} MB")
    else:
        print(f"❌ Model file not found")
        return False
    
    try:
        model = get_model()
        print("✅ Model loaded successfully")
        print(f"   Device: {next(model.model.parameters()).device}")
    except Exception as e:
        print(f"❌ Model loading failed: {e}")
        return False
    
    try:
        dummy_input = torch.randn(1, 3, 224, 224)
        if torch.cuda.is_available():
            dummy_input = dummy_input.cuda()
        with torch.no_grad():
            output = model.model(dummy_input)
        print("✅ Model inference test passed")
    except Exception as e:
        print(f"❌ Inference failed: {e}")
        return False
    
    print("\n✅ ALL CHECKS PASSED")
    return True

if __name__ == "__main__":
    verify_model()
