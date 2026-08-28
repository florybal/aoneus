import os
import shutil
import numpy as np
def construct_pose_matrix(pose_row):
    from scipy.spatial.transform import Rotation as R
    x, y, z, yaw, pitch, roll = pose_row
    rot = R.from_euler('zyx', [yaw, pitch, roll], degrees=False).as_matrix()
    pose = np.eye(4, dtype=np.float32)
    pose[:3, :3] = rot
    pose[:3, 3] = [x, y, z]
    return np.linalg.inv(pose)

def main():
    src_dir = "data/SonarCloud/synthetic_rgb/image"
    pose_dir = "data/SonarCloud/cluster/objects_pose_fixed8"
    
    terrains = ["more_ruggedy_terrain", "straighter_terrain", "straightest_terrain", "no_terrain"]
    objects = ["500lbs_data", "boat_data", "cube_data", "cylinder_data", "lshape_data", "mine_filtered_data", "mortar_shell_data", "pipe_data", "rectangle_data", "semisphere_data", "sphere_data", "stairs_data", "tire_data", "trapezoid_data", "triangle_data", "ushape_data", "uxo_big_data", "uxo_deformed_data", "uxo_small_data"]

    # Intrinsics
    H, W = 256, 96
    hfov, vfov = 60.0, 12.0
    fx = W / (2 * np.tan(np.radians(hfov) / 2))
    fy = H / (2 * np.tan(np.radians(vfov) / 2))
    cx, cy = W / 2, H / 2
    camera_mat = np.array([[fx, 0, cx, 0], [0, fy, cy, 0], [0, 0, 1, 0], [0, 0, 0, 1]], dtype=np.float32)
    scale_mat = np.eye(4, dtype=np.float32)

    files = [f for f in os.listdir(src_dir) if f.endswith(".png")]
    print(f"Total de imagens para organizar: {len(files)}")

    groups = {}
    for f in files:
        obj = next((o for o in objects if f.startswith(o)), None)
        if not obj: continue
        rem = f[len(obj)+1:]
        terr = next((t for t in terrains if rem.startswith(t)), None)
        if not terr: continue
        rem2 = rem[len(terr)+1:]
        parts = rem2.split("_")
        if len(parts) < 4: continue
        variation = parts[0] + "_" + parts[1] # e.g. moved_terrain0
        rot = parts[2] + "_" + parts[3] # e.g. rot0_0
        
        base_name = f.rsplit("_v", 1)[0]
        
        key = (obj, terr, variation, rot, base_name)
        if key not in groups:
            groups[key] = []
        groups[key].append(f)

    print(f"Total de grupos (orientações/poses): {len(groups)}")

    success_count = 0
    for key, img_files in groups.items():
        obj, terr, variation, rot, base_name = key
        target_dir = os.path.join("data/SonarCloud", obj, terr, variation, rot)
        image_target_dir = os.path.join(target_dir, "image")
        os.makedirs(image_target_dir, exist_ok=True)

        pose_path = os.path.join(pose_dir, base_name + ".npy")
        if not os.path.exists(pose_path):
            continue

        poses = np.load(pose_path, allow_pickle=True)
        
        cameras = {}
        valid_idx = 0

        # Sort images by view index (_v0, _v1, ...)
        img_files_sorted = sorted(img_files, key=lambda x: int(x.rsplit("_v", 1)[1].replace(".png", "")))

        for img_file in img_files_sorted:
            src_path = os.path.join(src_dir, img_file)
            dst_path = os.path.join(image_target_dir, img_file)
            shutil.move(src_path, dst_path)

            view_idx = int(img_file.rsplit("_v", 1)[1].replace(".png", ""))
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
            np.savez(os.path.join(target_dir, "cameras_sphere.npz"), **cameras)
            success_count += 1

    print(f"Reorganização concluída com sucesso! {success_count} diretórios organizados.")

if __name__ == "__main__":
    main()
