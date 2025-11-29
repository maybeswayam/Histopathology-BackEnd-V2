from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from PIL import Image
import os
import io
import logging
from src.model.model_loader import get_model
from src.model.gradcam_service import GradCAM, overlay_heatmap
import base64
import torch

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Histopathology Cancer Detection API",
    description="API for detecting cancer in histopathology images using CNN",
    version="1.0.0"
)

# Add CORS middleware for JavaScript frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure this properly for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize model on startup
model = None

@app.on_event("startup")
async def startup_event():
    global model
    try:
        model = get_model()
        logger.info("✅ Model initialized successfully")
    except Exception as e:
        logger.error(f"❌ Model initialization failed: {str(e)}")
        raise e

@app.get("/")
async def root():
    return {"message": "HistoAI Backend is running"}

@app.get("/health")
async def health_check():
    """Check if the API and model are healthy"""
    if model is None:
        return JSONResponse(
            status_code=503,
            content={"status": "error", "message": "Model not initialized"}
        )
    return {"status": "healthy", "model_loaded": True}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    """Standard prediction endpoint"""
    try:
        image_data = await file.read()
        image = Image.open(io.BytesIO(image_data)).convert('RGB')
        prediction_result = model.predict(image)
        return prediction_result
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/predict-with-gradcam")
async def predict_with_gradcam(file: UploadFile = File(...)):
    """Predict with Grad-CAM visualization using the standard approach from generate_gradcam.py"""
    try:
        image_data = await file.read()
        image = Image.open(io.BytesIO(image_data)).convert('RGB')
        
        # Get prediction first
        prediction_result = model.predict(image)
        
        # Prepare input tensor for Grad-CAM (requires gradient)
        device = next(model.model.parameters()).device
        input_tensor = model.transform(image).unsqueeze(0).to(device)
        input_tensor.requires_grad_(True)
        
        # Get target class index
        class_idx = 1 if prediction_result['prediction'] == 'malignant' else 0
        
        # Generate Grad-CAM using the standard approach
        gradcam_service = GradCAM(model.model)
        heatmap = gradcam_service(input_tensor, class_idx)
        
        # Create overlay visualization
        overlayed_image = overlay_heatmap(heatmap, image)
        
        # Convert to base64 for API response
        buffered = io.BytesIO()
        overlayed_image.save(buffered, format="PNG")
        heatmap_base64 = base64.b64encode(buffered.getvalue()).decode()
        
        return {
            **prediction_result,
            "heatmap": f"data:image/png;base64,{heatmap_base64}"
        }
    except Exception as e:
        logger.error(f"Grad-CAM prediction error: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))
