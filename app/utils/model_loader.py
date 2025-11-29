import os
import torch
import torchvision.transforms as transforms
from PIL import Image
import logging
from pathlib import Path
from torchvision.models import mobilenet_v2
from typing import Optional, Tuple

logger = logging.getLogger(__name__)

class ModelLoader:
    def __init__(self, model_path: str = "models/best_model.pth"):
        self.model_path = model_path
        self.model = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                              std=[0.229, 0.224, 0.225])
        ])
        self.classes = ['benign', 'malignant']
        self._load_model()

    def _load_model(self) -> None:
        """Loads the PyTorch model from the specified path."""
        try:
            # Create model directory if it doesn't exist
            os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
            
            # Initialize model architecture
            self.model = mobilenet_v2(pretrained=False)
            self.model.classifier[1] = torch.nn.Linear(
                in_features=self.model.classifier[1].in_features, 
                out_features=2
            )

            # Load weights if they exist
            if os.path.exists(self.model_path):
                logger.info(f"Loading model from {self.model_path}")
                state_dict = torch.load(self.model_path, map_location=self.device)
                self.model.load_state_dict(state_dict)
                logger.info("Model loaded successfully")
            else:
                logger.warning(f"Model file not found at {self.model_path}")
                logger.warning("Using initialized model without trained weights")

            self.model.to(self.device)
            self.model.eval()

        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            raise

    def preprocess_image(self, image: Image.Image) -> torch.Tensor:
        """Preprocesses the input image for model inference."""
        try:
            # Convert grayscale to RGB if needed
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            # Apply transformations
            tensor = self.transform(image)
            tensor = tensor.unsqueeze(0)  # Add batch dimension
            return tensor.to(self.device)

        except Exception as e:
            logger.error(f"Error preprocessing image: {str(e)}")
            raise

    def predict(self, image: Image.Image) -> Tuple[str, float, dict]:
        """
        Performs inference on the input image.
        Returns: (prediction_class, confidence_score, probabilities_dict)
        """
        try:
            if self.model is None:
                raise RuntimeError("Model not initialized")

            # Preprocess image
            tensor = self.preprocess_image(image)

            # Perform inference
            with torch.no_grad():
                outputs = self.model(tensor)
                probabilities = torch.softmax(outputs, dim=1)[0]

            # Get prediction and confidence
            pred_idx = torch.argmax(probabilities).item()
            confidence = probabilities[pred_idx].item()
            pred_class = self.classes[pred_idx]

            # Create probabilities dictionary
            probs_dict = {
                cls: prob.item() 
                for cls, prob in zip(self.classes, probabilities)
            }

            return pred_class, confidence, probs_dict

        except Exception as e:
            logger.error(f"Error during prediction: {str(e)}")
            raise

# Global model instance
model_loader: Optional[ModelLoader] = None

def get_model_loader() -> ModelLoader:
    """Returns a singleton instance of ModelLoader."""
    global model_loader
    if model_loader is None:
        model_loader = ModelLoader()
    return model_loader