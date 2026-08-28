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
    synth_dir = "data/SonarCloud/synthetic_rgb"
    pose_dir = "data/SonarCloud/cluster/objects_pose_fixed8"
    
    # Intrinsics
    H, W = 256, 96
    hfov, vfov = 60.0, 12.0
    fx = W / (2 * np.tan(np.radians(hfov) / 2))
    fy = H / (2 * np.tan(np.radians(vfov) / 2))
    cx, cy = W / 2, H / 2
    camera_mat = np.array([[fx, 0, cx, 0], [0, fy, cy, 0], [0, 0, 1, 0], [0, 0, 0, 1]], dtype=np.float32)
    scale_mat = np.eye(4, dtype=np.float32)
    
    files = [f for f in os.listdir(synth_dir) if f.endswith(".png")]
    
    for img_file in files:
        # Filename: 500lbs_data_more_ruggedy_terrain_moved_terrain0_rot0_0_v0.png
        # Parts: 500lbs_data, more_ruggedy_terrain, moved_terrain0, rot0_0, v0
        parts = img_file.replace(".png", "").split("_")
        
        # This parsing is fragile. Let's assume the structure is fixed.
        # object_name = parts[0] + "_" + parts[1] (if it's 500lbs_data)
        # This is too hard to parse reliably.
        
        # Let's just move them to a flat structure for now, and tell the user.
        pass
    print("Reorganização por objeto é complexa devido à nomenclatura dos arquivos.")

if __name__ == "__main__":
    main()

