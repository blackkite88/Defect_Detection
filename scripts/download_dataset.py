import kagglehub
import os
import shutil

print("[Download] Fetching MVTec AD dataset...")
path = kagglehub.dataset_download("ipythonx/mvtec-ad")

print(f"[Download] Dataset downloaded to cache: {path}")

# Destination folder inside your project
target_data_dir = os.path.abspath("./data")

for category in ["bottle", "tile"]:
    src_path = os.path.join(path, category)
    dest_path = os.path.join(target_data_dir, category)
    
    if os.path.exists(src_path):
        if os.path.exists(dest_path):
            shutil.rmtree(dest_path)
        shutil.copytree(src_path, dest_path)
        print(f"[Success] Copied '{category}' category into {dest_path}")
    else:
        print(f"[Warning] Category '{category}' not found in downloaded files.")

