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
    rgb_dir = "data/SonarCloud/synthetic_rgb"
    image_dir = os.path.join(rgb_dir, "image")
    os.makedirs(image_dir, exist_ok=True)
    
    pose_dir = "data/SonarCloud/cluster/objects_pose_fixed8"
    
    # Intrinsics
    H, W = 256, 96
    hfov, vfov = 60.0, 12.0
    fx = W / (2 * np.tan(np.radians(hfov) / 2))
    fy = H / (2 * np.tan(np.radians(vfov) / 2))
    cx, cy = W / 2, H / 2
    
    camera_mat = np.array([
        [fx, 0, cx, 0],
        [0, fy, cy, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1]
    ], dtype=np.float32)
    scale_mat = np.eye(4, dtype=np.float32)
    
    cameras = {}
    
    # List all png files in rgb_dir (not in image/ yet)
    files = [f for f in os.listdir(rgb_dir) if f.endswith(".png")]
    print(f"Total de imagens encontradas: {len(files)}")
    
    valid_idx = 0
    for img_file in sorted(files):
        src_path = os.path.join(rgb_dir, img_file)
        dst_path = os.path.join(image_dir, img_file)
        
        # Mover para image/
        if not os.path.exists(dst_path):
            shutil.move(src_path, dst_path)
            
        parts = img_file.replace(".png", "").rsplit("_v", 1)
        if len(parts) != 2:
            continue
            
        base_name, view_str = parts[0], parts[1]
        try:
            view_idx = int(view_str)
        except ValueError:
            continue
            
        pose_path = os.path.join(pose_dir, base_name + ".npy")
        if not os.path.exists(pose_path):
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
        
    np.savez(os.path.join(rgb_dir, "cameras_sphere.npz"), **cameras)
    print(f"cameras_sphere.npz gerado com sucesso com {valid_idx} câmeras em {image_dir}!")

if __name__ == "__main__":
    main()
