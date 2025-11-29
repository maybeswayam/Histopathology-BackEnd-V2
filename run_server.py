#!/usr/bin/env python3
"""
Simple script to run the FastAPI server
"""

import uvicorn
import os
import sys

def main():
    """Run the FastAPI server"""
    # Check if model directory exists
    if not os.path.exists("models"):
        print("Creating models directory...")
        os.makedirs("models")
    
    # Check for model file (either naming convention)
    model_files = ["models/best_model.pth", "models/model_best.pth"]
    model_found = any(os.path.exists(path) for path in model_files)
    
    if not model_found:
        print("Warning: No model file found. Looking for either:")
        for path in model_files:
            print(f"- {path}")
        print("\nThe server will start and attempt to find a model in alternative locations.")
    
    # Run the server
    print("Starting Histopathology Cancer Detection API server...")
    print("Server will be available at: http://localhost:8000")
    print("API documentation: http://localhost:8000/docs")
    print("Press Ctrl+C to stop the server")
    
    try:
        uvicorn.run(
            "app:app",
            host="0.0.0.0",
            port=8000,
            reload=True,
            log_level="info"
        )
    except KeyboardInterrupt:
        print("\nServer stopped by user")
    except Exception as e:
        print(f"Error starting server: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
