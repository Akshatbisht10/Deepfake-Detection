import os
import shutil
import random
import glob
from tqdm import tqdm

# Configuration
DATASET_DIR = "dataset_faces"
OUTPUT_DIR = "dataset_split"
SPLIT_RATIO = (0.7, 0.15, 0.15) # Train, Val, Test
SEED = 42

def create_splits():
    random.seed(SEED)
    
    # Create output directories
    for split in ['train', 'val', 'test']:
        for label in ['real', 'fake']:
            os.makedirs(os.path.join(OUTPUT_DIR, split, label), exist_ok=True)
            
    # Process each class
    for label in ['real', 'fake']:
        source_dir = os.path.join(DATASET_DIR, label)
        if not os.path.exists(source_dir):
            print(f"Warning: {source_dir} not found. Skipping.")
            continue
            
        # Get all image files
        images = glob.glob(os.path.join(source_dir, "*.jpg"))
        random.shuffle(images)
        
        n_total = len(images)
        n_train = int(n_total * SPLIT_RATIO[0])
        n_val = int(n_total * SPLIT_RATIO[1])
        # n_test is the rest
        
        train_imgs = images[:n_train]
        val_imgs = images[n_train:n_train+n_val]
        test_imgs = images[n_train+n_val:]
        
        print(f"Processing {label}: Total={n_total}, Train={len(train_imgs)}, Val={len(val_imgs)}, Test={len(test_imgs)}")
        
        # Helper to copy images
        def copy_images(image_list, split_name):
            for img_path in tqdm(image_list, desc=f"Copying {label} to {split_name}"):
                filename = os.path.basename(img_path)
                dst = os.path.join(OUTPUT_DIR, split_name, label, filename)
                shutil.copy(img_path, dst)
                
        copy_images(train_imgs, 'train')
        copy_images(val_imgs, 'val')
        copy_images(test_imgs, 'test')

if __name__ == "__main__":
    if not os.path.exists(DATASET_DIR):
        print(f"Error: {DATASET_DIR} does not exist. Run preprocess.py first.")
    else:
        create_splits()
