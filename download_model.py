import os
import gdown
import sys

MODEL_DIR = "./models"
MODEL_PATH = f"{MODEL_DIR}/model_best.pth"
GDRIVE_FILE_ID = "1gNl6HIptmbvhhvqL0iZoFaB7aD1s_3_M"

def download_model():
    os.makedirs(MODEL_DIR, exist_ok=True)
    if os.path.exists(MODEL_PATH):
        print(f"✅ Model already exists at {MODEL_PATH}")
        file_size_mb = os.path.getsize(MODEL_PATH) / (1024 * 1024)
        print(f"   Model size: {file_size_mb:.2f} MB")
        return True
    print(f"📥 Downloading large model (655MB) from Google Drive...")
    try:
        url = f"https://drive.google.com/uc?id={GDRIVE_FILE_ID}"
        gdown.download(url, MODEL_PATH, quiet=False)
        file_size_mb = os.path.getsize(MODEL_PATH) / (1024 * 1024)
        print(f"✅ Model downloaded successfully! Size: {file_size_mb:.2f} MB")
        return True
    except Exception as e:
        print(f"❌ Error downloading model: {e}")
        return False

if __name__ == "__main__":
    success = download_model()
    sys.exit(0 if success else 1)
