
import os
import shutil
from sklearn.model_selection import train_test_split

# Define paths
data_dir = 'fer2013/train'
val_dir = 'fer2013/val'
os.makedirs(val_dir, exist_ok=True)

# Loop through each class directory
for class_name in os.listdir(data_dir):
    class_path = os.path.join(data_dir, class_name)
    if not os.path.isdir(class_path):
        continue

    # List all images
    images = os.listdir(class_path)
    train_imgs, val_imgs = train_test_split(images, test_size=0.1, random_state=42)

    # Create val class folder
    val_class_path = os.path.join(val_dir, class_name)
    os.makedirs(val_class_path, exist_ok=True)

    # Copy validation images
    for img in val_imgs:
        src = os.path.join(class_path, img)
        dst = os.path.join(val_class_path, img)
        shutil.copy2(src, dst)

print("Validation set created at:", val_dir)
