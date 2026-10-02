import urllib.request
import zipfile
import os
import shutil

# Direct link to just the MVTec Bottle category (~150 MB)
url = "https://github.com/doyupko/mvtec-ad-dataset/raw/master/bottle.zip"
zip_path = "bottle.zip"
extract_dir = "data/bottle"

os.makedirs("data", exist_ok=True)

print("Downloading lightweight MVTec Bottle sample (~150 MB)...")
try:
    urllib.request.urlretrieve("https://huggingface.co/datasets/briaai/MVTec-AD/resolve/main/bottle.zip", zip_path)
    print("Extracting files...")
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall("data/")
    os.remove(zip_path)
    print("Done! Bottle category saved to ./data/bottle")
except Exception as e:
    print(f"Direct download failed: {e}")
    print("You can let the Kaggle download finish, or manually place test images into ./data/")

