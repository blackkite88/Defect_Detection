import urllib.request
import os
import tarfile
import shutil

tar_path = "bottle.tar.xz"
# Direct mirror link for ONLY the MVTec Bottle category (~158 MB)
direct_url = "https://www.mydrive.ch/shares/38536/3830184030e49fe74747669442f0f282/download/420937370-1629951468/bottle.tar.xz"

os.makedirs("data", exist_ok=True)

if not os.path.exists("data/bottle"):
    print("[1/3] Downloading MVTec 'bottle' archive (~158 MB)...")
    req = urllib.request.Request(direct_url, headers={'User-Agent': 'Mozilla/5.0'})
    
    with urllib.request.urlopen(req) as response, open(tar_path, 'wb') as out_file:
        shutil.copyfileobj(response, out_file)
        
    print("[2/3] Extracting archive to ./data/bottle...")
    with tarfile.open(tar_path, "r:xz") as tar:
        tar.extractall(path="data")
                
    print("[3/3] Cleaning up temporary archive...")
    if os.path.exists(tar_path):
        os.remove(tar_path)
        
    print("✅ Success! MVTec 'bottle' dataset extracted at ./data/bottle")
else:
    print("✅ './data/bottle' already exists!")

