import os
import shutil
import numpy as np
from scipy.spatial.transform import Rotation as R

def construct_pose_matrix(pose_row):
    x, y, z, yaw, pitch, roll = pose_row
    rot = R.from_euler('zyx', [yaw, pitch, roll], degrees=False).as_matrix()
    pose = np.eye(4, dtype=np.float32)
    pose[:3, :3] = rot
    pose[:3, 3] = [x, y, z]
    return np.linalg.inv(pose)

def main():
    src_dir = "data/SonarCloud/synthetic_rgb/image"
    pose_dir = "data/SonarCloud/cluster/objects_pose_fixed8"
    
    # Intrinsics
    H, W = 256, 96
    hfov, vfov = 60.0, 12.0
    fx = W / (2 * np.tan(np.radians(hfov) / 2))
    fy = H / (2 * np.tan(np.radians(vfov) / 2))
    cx, cy = W / 2, H / 2
    camera_mat = np.array([[fx, 0, cx, 0], [0, fy, cy, 0], [0, 0, 1, 0], [0, 0, 0, 1]], dtype=np.float32)
    scale_mat = np.eye(4, dtype=np.float32)
    
    files = [f for f in os.listdir(src_dir) if f.endswith(".png")]
    
    # Group files by (object, terrain, variation)
    groups = {}
    for img_file in files:
        # Filename: 500lbs_data_more_ruggedy_terrain_moved_terrain0_rot0_0_v0.png
        parts = img_file.replace(".png", "").split("_")
        
        # Assuming structure: object_name, terrain, variation, ...
        # This is still a bit fragile, but let's try to be more robust.
        # Maybe we can look at the existing folder structure in data/SonarCloud/boat_data/
        # boat_data / more_ruggedy_terrain / moved_terrain0
        
        # Let's try to find the object name from the list of folders in data/SonarCloud/
        # 500lbs_data, boat_data, cube_data, ...
        
        object_name = None
        for obj in ["500lbs_data", "boat_data", "cube_data", "cylinder_data", "lshape_data", "mine_filtered_data", "mortar_shell_data", "pipe_data", "rectangle_data", "semisphere_data", "sphere_data", "stairs_data", "tire_data", "trapezoid_data", "triangle_data", "ushape_data", "uxo_big_data", "uxo_deformed_data", "uxo_small_data"]:
            if img_file.startswith(obj):
                object_name = obj
                break
        
        if object_name is None:
            print(f"Could not identify object for {img_file}")
            continue
            
        # Remaining parts: more_ruggedy_terrain_moved_terrain0_rot0_0_v0
        remaining = img_file.replace(object_name + "_", "", 1).replace(".png", "")
        parts = remaining.split("_")
        
        # This is still tricky. Let's look at the boat_data structure again:
        # boat_data / more_ruggedy_terrain / moved_terrain0
        
        # Maybe we can just use the first few parts.
        terrain = parts[0] + "_" + parts[1] # more_ruggedy_terrain
        variation = parts[2] # moved_terrain0
        
        key = (object_name, terrain, variation)
        if key not in groups:
            groups[key] = []
        groups[key].append(img_file)
        
    for key, img_files in groups.items():
        object_name, terrain, variation = key
        dst_dir = os.path.join("data/SonarCloud", object_name, terrain, variation, "image")
        os.makedirs(dst_dir, exist_ok=True)
        
        cameras = {}
        valid_idx = 0
        
        for img_file in sorted(img_files):
            src_path = os.path.join(src_dir, img_file)
            dst_path = os.path.join(dst_dir, img_file)
            shutil.move(src_path, dst_path)
            
            # Generate camera
            # Filename: 500lbs_data_more_ruggedy_terrain_moved_terrain0_rot0_0_v0.png
            # Parts: 500lbs_data, more_ruggedy_terrain, moved_terrain0, rot0_0, v0
            # We need the base name for pose lookup: 500lbs_data_more_ruggedy_terrain_moved_terrain0_rot0_0
            
            # This is tricky. Let's look at how generate_cameras.py did it.
            # parts = img_file.replace(".png", "").rsplit("_v", 1)
            # base_name = parts[0]
            
            base_name = img_file.replace(".png", "").rsplit("_v", 1)[0]
            view_idx = int(img_file.replace(".png", "").rsplit("_v", 1)[1])
            
            pose_path = os.path.join(pose_dir, base_name + ".npy")
            if not os.path.exists(pose_path):
                # Try to find the pose file. Maybe it's in a different folder?
                # The pose_dir is data/SonarCloud/cluster/objects_pose_fixed8
                continue
                
            poses = np.load(pose_path)
            if view_idx >= len(poses):
                continue
                
            pose_row = poses[view_idx]
            extrinsic = construct_pose_matrix(pose_row)
            world_mat = camera_mat @ extrinsic
            
            cameras[f"world_mat_{valid_idx}"] = world_mat
            cameras[f"world_mat_inv_{valid_idx}"] = np.linalg.inv(world_mat)
            cameras[f"camera_mat_{valid_idx}"] = camera_mat
            cameras[f"camera_mat_inv_{valid_idx}"] = np.linalg.inv(camera_mat)
            cameras[f"scale_mat_{valid_idx}"] = scale_mat
            cameras[f"scale_mat_inv_{valid_idx}"] = np.linalg.inv(scale_mat)
            
            valid_idx += 1
            
        if valid_idx > 0:
            np.savez(os.path.join("data/SonarCloud", object_name, terrain, variation, "cameras_sphere.npz"), **cameras)
            print(f"Generated cameras_sphere.npz for {object_name}/{terrain}/{variation} with {valid_idx} cameras.")

if __name__ == "__main__":
    main()