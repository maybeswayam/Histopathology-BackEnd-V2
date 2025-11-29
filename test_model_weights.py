import torch
from src.model.model_loader import get_model
from src.model.model import get_model as get_architecture
from PIL import Image
import numpy as np

print("="*60)
print("TESTING MODEL WEIGHT LOADING")
print("="*60)

# Step 1: Load the trained model
print("\n1. Loading model through model_loader...")
trained_model = get_model()
print("✅ Model loaded")

# Step 2: Create a fresh random model
print("\n2. Creating fresh random model...")
random_model = get_architecture(name="mobilenet_v2", num_classes=2, pretrained=False)
random_model.eval()
print("✅ Fresh model created")

# Step 3: Compare first layer weights
print("\n3. Comparing weights...")
trained_first_layer = list(trained_model.model.parameters())[0]
random_first_layer = list(random_model.parameters())[0]

weight_diff = torch.abs(trained_first_layer - random_first_layer).mean().item()
print(f"   Average weight difference: {weight_diff:.6f}")

if weight_diff < 0.001:
    print("   ❌ PROBLEM: Weights are nearly identical to random init!")
    print("   The trained weights are NOT being loaded properly.")
else:
    print("   ✅ Weights are different - trained weights are loaded!")

# Step 4: Test prediction consistency
print("\n4. Testing prediction consistency...")
# Create a simple test image
test_image = Image.new('RGB', (224, 224), color=(128, 128, 128))

prediction = trained_model.predict(test_image)
print(f"   Prediction: {prediction['prediction']}")
print(f"   Confidence: {prediction['confidence']:.4f}")
print(f"   Probabilities: benign={prediction['probabilities']['benign']:.4f}, malignant={prediction['probabilities']['malignant']:.4f}")

# Run it twice to check consistency
prediction2 = trained_model.predict(test_image)
if abs(prediction['confidence'] - prediction2['confidence']) < 0.0001:
    print("   ✅ Predictions are consistent")
else:
    print("   ❌ Predictions are inconsistent (shouldn't happen)")

print("\n" + "="*60)
print("CONCLUSION:")
print("="*60)
if weight_diff < 0.001:
    print("❌ MODEL WEIGHTS ARE NOT LOADED - still using random weights!")
else:
    print("✅ Model weights are loaded correctly")
