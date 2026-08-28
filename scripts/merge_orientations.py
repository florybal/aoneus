import os
import shutil
import numpy as np
from glob import glob

def merge_orientations():
    base_dir = "data/SonarCloud/boat_data/no_terrain/moved_terrain0"
    output_dir = os.path.join(base_dir, "all_orientations")
    output_image_dir = os.path.join(output_dir, "image")
    output_depth_dir = os.path.join(output_dir, "depth")
    output_sonar_dir = os.path.join(output_dir, "sonar_c")
    
    os.makedirs(output_image_dir, exist_ok=True)
    os.makedirs(output_depth_dir, exist_ok=True)
    os.makedirs(output_sonar_dir, exist_ok=True)
    
    orientations = sorted([d for d in os.listdir(base_dir) if d.startswith("orientation_")])
    
    all_cameras = {}
    total_images = 0
    
    for orient in orientations:
        orient_dir = os.path.join(base_dir, orient)
        image_dir = os.path.join(orient_dir, "image")
        depth_dir = os.path.join(orient_dir, "depth")
        sonar_dir = os.path.join(orient_dir, "sonar_c")
        cameras_path = os.path.join(orient_dir, "cameras_sphere.npz")
        
        if not os.path.exists(cameras_path):
            print(f"Skipping {orient}, no cameras_sphere.npz found.")
            continue
            
        cameras = np.load(cameras_path)
        image_files = sorted(glob(os.path.join(image_dir, "*.png")))
        
        # Assuming images, depth files, and sonar files are sorted and correspond to each other
        depth_files = sorted(glob(os.path.join(depth_dir, "*")))
        sonar_files = sorted(glob(os.path.join(sonar_dir, "*")))
        
        for i, img_path in enumerate(image_files):
            # New name to ensure uniqueness
            new_name = f"{orient}_{os.path.basename(img_path)}"
            shutil.copy(img_path, os.path.join(output_image_dir, new_name))
            
            # Copy depth files if they exist
            if i < len(depth_files):
                depth_path = depth_files[i]
                new_depth_name = f"{orient}_{os.path.basename(depth_path)}"
                shutil.copy(depth_path, os.path.join(output_depth_dir, new_depth_name))
            
            # Copy sonar files if they exist
            if i < len(sonar_files):
                sonar_path = sonar_files[i]
                new_sonar_name = f"{orient}_{os.path.basename(sonar_path)}"
                shutil.copy(sonar_path, os.path.join(output_sonar_dir, new_sonar_name))
            
            # Copy camera matrices
            key_suffix = f"_{i}"
            for key in cameras.files:
                if key.endswith(key_suffix):
                    base_key = key.rsplit("_", 1)[0]
                    new_key = f"{base_key}_{total_images}"
                    all_cameras[new_key] = cameras[key]
            
            total_images += 1
            
    np.savez(os.path.join(output_dir, "cameras_sphere.npz"), **all_cameras)
    print(f"Merged {total_images} images, depth files, and sonar files into {output_dir}")

if __name__ == "__main__":
    merge_orientations()
