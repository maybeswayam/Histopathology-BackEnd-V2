from src.model.model_loader import get_model

try:
    model = get_model()
    print("✅ Model loaded successfully and set to eval mode.")
    print(model)
except Exception as e:
    print("❌ Model loading failed:", str(e))