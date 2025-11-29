import shutil
import os

source = r"C:\Users\alexm\Downloads\mini project\vansh-model\models\models\checkpoint.pth"
dest = r"C:\Users\alexm\Downloads\mini project\uday-backend\models\model_best.pth"

print(f"Copying from: {source}")
print(f"To: {dest}")

if os.path.exists(source):
    shutil.copy(source, dest)
    print("✅ Copy successful!")
else:
    print("❌ Source file not found!")
