import torch
import os
from typing import Dict, Any, Optional
from pathlib import Path

def save_checkpoint(state: Dict[str, Any], is_best: bool, save_dir: str):
    """
    Save model checkpoint
    
    Args:
        state: Model state dictionary
        is_best: Whether this is the best model so far
        save_dir: Directory to save checkpoints
    """
    os.makedirs(save_dir, exist_ok=True)
    
    # Save regular checkpoint
    checkpoint_path = os.path.join(save_dir, 'checkpoint.pth')
    torch.save(state, checkpoint_path)
    
    # Save best model
    if is_best:
        best_path = os.path.join(save_dir, 'best_model.pth')
        torch.save(state, best_path)
        print(f"New best model saved to {best_path}")

def load_checkpoint(checkpoint_path: str) -> Optional[Dict[str, Any]]:
    """
    Load model checkpoint
    
    Args:
        checkpoint_path: Path to checkpoint file
    
    Returns:
        Checkpoint dictionary or None if loading failed
    """
    try:
        if os.path.exists(checkpoint_path):
            checkpoint = torch.load(checkpoint_path, map_location='cpu')
            return checkpoint
        else:
            print(f"Checkpoint not found at {checkpoint_path}")
            return None
    except Exception as e:
        print(f"Error loading checkpoint: {e}")
        return None

def get_project_root() -> Path:
    """
    Get the project root directory
    
    Returns:
        Path to the project root
    """
    # Assuming we're in src/model/utils.py, go up two levels
    return Path(__file__).parent.parent.parent

def load_trained_model(device: str = 'cpu') -> torch.nn.Module:
    """
    Load the trained model for inference
    
    Args:
        device: Device to load the model on ('cpu' or 'cuda')
        
    Returns:
        Loaded PyTorch model in eval mode
        
    Raises:
        FileNotFoundError: If model file is not found
        RuntimeError: If model loading fails
    """
    root_dir = get_project_root()
    model_path = root_dir / 'models' / 'model_best.pth'
    
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found at {model_path}")

    try:
        model = torch.load(str(model_path), map_location=device)
        model.eval()  # Set to evaluation mode
        print("✅ Loaded trained model successfully and set to eval mode.")
        return model
    except Exception as e:
        raise RuntimeError(f"Failed to load model: {str(e)}")
