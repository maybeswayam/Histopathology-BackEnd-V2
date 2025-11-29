import torch
import torchvision.transforms as transforms
from pathlib import Path

class HistoPathModel:
    def __init__(self, device='cpu'):
        self.device = device
        self.model = self._load_model()
        self.transform = self._get_transforms()

    def _find_model_file(self):
        """Find the model file in various possible locations"""
        # Primary model path
        primary_path = Path('./models/model_best.pth')
        if primary_path.exists():
            return primary_path
        
        possible_names = ['model_best.pth', 'best_model.pth']
        possible_paths = [
            Path.cwd(),  # Current working directory
            Path.cwd() / 'models',  # models/ in current directory
            Path(__file__).parent.parent.parent / 'models',  # project_root/models/
            Path(__file__).parent.parent.parent.parent / 'vansh-model' / 'models'  # Check vansh-model folder
        ]
        
        for path in possible_paths:
            if path.exists():
                for name in possible_names:
                    model_path = path / name
                    if model_path.exists():
                        return model_path
        
        raise FileNotFoundError(
            "Model file not found. Searched for model_best.pth or best_model.pth in:\n" +
            "\n".join(f"- {p}" for p in possible_paths)
        )

    def _load_model(self):
        """Load the model from disk"""
        try:
            model_path = self._find_model_file()
            print(f"✅ Found model at: {model_path}")

            from src.model.model import ResNet50_VGG16_Fusion
            
            # Load checkpoint
            checkpoint = torch.load(str(model_path), map_location=self.device)
            
            # Initialize model architecture
            model = ResNet50_VGG16_Fusion(num_classes=2, pretrained=False)
            
            # Determine if the checkpoint is a state_dict or a full checkpoint dict
            if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                state_dict = checkpoint['model_state_dict']
            elif isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
                state_dict = checkpoint['state_dict']
            else:
                state_dict = checkpoint # Assume the file is the state_dict itself

            # Load the state dict
            model.load_state_dict(state_dict)
                
            model.eval()
            return model
        except Exception as e:
            print(f"❌ Error loading model: {str(e)}")
            raise

    def _get_transforms(self):
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])

    @torch.no_grad()
    def predict(self, image):
        """
        Make prediction on a single image
        
        Args:
            image: PIL Image object
            
        Returns:
            dict: Prediction results with class and confidence
        """
        # Prepare image
        tensor = self.transform(image).unsqueeze(0)
        
        # Get prediction
        outputs = self.model(tensor)
        probs = torch.nn.functional.softmax(outputs, dim=1)
        confidence, predicted = torch.max(probs, 1)
        
        # Get all class probabilities
        class_probs = probs[0].tolist()
        
        return {
            "prediction": "malignant" if predicted.item() == 1 else "benign",
            "confidence": confidence.item(),
            "probabilities": {
                "benign": class_probs[0],
                "malignant": class_probs[1]
            }
        }

# Global model instance
_model_instance = None

def get_model(device='cpu'):
    """
    Get or create the model instance (singleton pattern)
    """
    global _model_instance
    if _model_instance is None:
        _model_instance = HistoPathModel(device)
    return _model_instance
