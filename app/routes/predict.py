from fastapi import APIRouter, UploadFile, HTTPException
from PIL import Image
import io
from typing import Dict, Union
from ..utils.model_loader import get_model_loader
import logging

logger = logging.getLogger(__name__)
router = APIRouter()

@router.post("/predict")
async def predict_image(file: UploadFile) -> Dict[str, Union[str, float, dict]]:
    """
    Endpoint for cancer detection prediction from histopathology images.
    
    Parameters:
        file: UploadFile - The image file to analyze
        
    Returns:
        dict containing:
        - prediction: str (benign/malignant)
        - confidence: float (0-1)
        - probabilities: dict of class probabilities
        - file_info: dict with filename and size
    """
    try:
        # Validate file type
        if not file.content_type.startswith("image/"):
            raise HTTPException(
                status_code=400, 
                detail="File must be an image"
            )

        # Read image file
        contents = await file.read()
        image = Image.open(io.BytesIO(contents))

        # Get model prediction
        model_loader = get_model_loader()
        prediction, confidence, probabilities = model_loader.predict(image)

        # Prepare response
        return {
            "prediction": prediction,
            "confidence": confidence,
            "probabilities": probabilities,
            "file_info": {
                "filename": file.filename,
                "content_type": file.content_type,
                "size": len(contents)
            }
        }

    except Exception as e:
        logger.error(f"Error processing prediction request: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Error processing image: {str(e)}"
        )