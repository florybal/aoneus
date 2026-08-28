import os
import numpy as np
import scipy.io as sio
import cv2

def process_depth_and_pose():
    depth_dir = "data/SonarCloud/cluster/objects_depth_fixed8"
    pose_dir = "data/SonarCloud/cluster/objects_pose_fixed8"
    output_dir = "data/SonarCloud/synthetic_rgb"
    os.makedirs(output_dir, exist_ok=True)
    
    depth_files = sorted([f for f in os.listdir(depth_dir) if f.endswith(".mat")])
    print(f"Total de arquivos de depth encontrados: {len(depth_files)}")
    
    for filename in depth_files:
        base_name = filename.replace(".mat", "")
        depth_path = os.path.join(depth_dir, filename)
        pose_path = os.path.join(pose_dir, base_name + ".npy")
        
        if not os.path.exists(pose_path):
            continue
            
        try:
            mat = sio.loadmat(depth_path)
            if 'Z' not in mat:
                continue
            Z = mat['Z'] # shape (8, 256, 256)
            
            poses = np.load(pose_path, allow_pickle=True) # shape (8, 6) or similar
            
            for i in range(Z.shape[0]):
                depth_map = Z[i]
                
                # Normalizar depth para 0-255 (ignorando zeros ou valores inválidos)
                valid_mask = (depth_map > 0.0) & np.isfinite(depth_map)
                if not np.any(valid_mask):
                    img = np.zeros((depth_map.shape[0], depth_map.shape[1], 3), dtype=np.uint8)
                else:
                    d_min = depth_map[valid_mask].min()
                    d_max = depth_map[valid_mask].max()
                    
                    if d_max > d_min:
                        norm_depth = np.clip((depth_map - d_min) / (d_max - d_min), 0.0, 1.0)
                    else:
                        norm_depth = np.zeros_like(depth_map)
                        
                    norm_depth[~valid_mask] = 0.0
                    img_gray = (norm_depth * 255).astype(np.uint8)
                    
                # Aplicar colormap ou converter para 3 canais RGB
                img_rgb = cv2.applyColorMap(img_gray, cv2.COLORMAP_TURBO)
                img_rgb[~valid_mask] = 0 # fundo preto
                
                out_filename = f"{base_name}_v{i}.png"
                cv2.imwrite(os.path.join(output_dir, out_filename), img_rgb)
                
        except Exception as e:
            print(f"Erro ao processar {base_name}: {e}")

    print("Processamento de depth concluído com sucesso!")

if __name__ == "__main__":
    process_depth_and_pose()


