"""Download PlantVillage subset from Kaggle.

Requires Kaggle API credentials (~/.kaggle/kaggle.json).
"""
import os, subprocess, zipfile, shutil

dataset = "plantvillage-dataset/plantvillage-dataset"
zip_path = "plantvillage.zip"

print("Downloading dataset from Kaggle...")
subprocess.check_call(["kaggle", "datasets", "download", "-d", dataset, "-p", "."])

print("Unzipping...")
with zipfile.ZipFile(zip_path, "r") as zf:
    zf.extractall("pv")

# Move tomato classes to data/train/
mapping = {
    "Tomato___healthy": "Tomato_Healthy",
    "Tomato___Bacterial_spot": "Tomato_Bacterial_spot",
    "Tomato___Leaf_Mold": "Tomato_Leaf_Mold",
    "Tomato___Early_blight": "Tomato_Early_blight",
    "Tomato___Septoria_leaf_spot": "Tomato_Septoria_leaf_spot",
    "Tomato___Late_blight": "Tomato_Late_blight",
    "Tomato___Leaf_Curl_Virus": "Tomato_Leaf_Curl_Virus"
}

os.makedirs("data/train", exist_ok=True)
for src, dst in mapping.items():
    src_dir = os.path.join("pv", src)
    if not os.path.isdir(src_dir):
        continue
    dst_dir = os.path.join("data/train", dst)
    shutil.move(src_dir, dst_dir)

shutil.rmtree("pv")
os.remove(zip_path)
print("Dataset ready under data/train/")
