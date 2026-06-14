import os
import shutil
from pathlib import Path

def prepare_refuge_classification(source_dir, dest_dir):
    """
    Organizes the REFUGE dataset (where filenames start with 'g' or 'n') 
    into a classification structure: Glaucoma/ and Normal/
    """
    print(f"Preparing classification dataset from {source_dir} to {dest_dir}")
    
    for split in ['train', 'val', 'test']:
        split_src = os.path.join(source_dir, split, 'images')
        
        if not os.path.exists(split_src):
            continue
            
        glaucoma_dir = os.path.join(dest_dir, split, 'Glaucoma')
        normal_dir = os.path.join(dest_dir, split, 'Normal')
        
        os.makedirs(glaucoma_dir, exist_ok=True)
        os.makedirs(normal_dir, exist_ok=True)
        
        # Move/copy files based on prefix
        for filename in os.listdir(split_src):
            src_path = os.path.join(split_src, filename)
            if filename.lower().startswith('g'):
                dest_path = os.path.join(glaucoma_dir, filename)
            elif filename.lower().startswith('n'):
                dest_path = os.path.join(normal_dir, filename)
            else:
                continue
                
            # Use copy2 to preserve metadata, or symlink if supported
            if not os.path.exists(dest_path):
                shutil.copy2(src_path, dest_path)
                
    print("Done organizing classification data!")

if __name__ == "__main__":
    src = "EyeVisionAI/datasets/REFUGE/REFUGE2"
    dst = "EyeVisionAI/datasets/ClassificationData"
    prepare_refuge_classification(src, dst)
