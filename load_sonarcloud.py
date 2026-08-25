import os
import torch
import numpy as np
import cv2
import json
from tqdm import tqdm
from scipy.spatial.transform import Rotation as Rot

def parse_pose_from_path(path):
    """
    Extrai pose (posição e orientação) a partir do caminho da imagem.
    Exemplo: .../moved_terrain0/rot45_0/sonar_c/FLSc_10.jpg
    """
    parts = path.split(os.sep)
    # Encontra a parte que contém 'moved_terrain' e 'rot'
    moved = None
    rot = None
    for p in parts:
        if 'moved_terrain' in p:
            moved = p  # ex: 'moved_terrain0'
        if 'rot' in p and '_' in p:
            rot = p    # ex: 'rot45_0'
    if moved is None or rot is None:
        return None

    # Extrai número do terreno (0, 1, 2)
    moved_idx = int(moved.replace('moved_terrain', ''))
    # Extrai ângulos de rotação (ex: rot45_0 -> 45, 0)
    rot_parts = rot.replace('rot', '').split('_')
    rot_x = int(rot_parts[0])
    rot_y = int(rot_parts[1])

    # Define posição do sensor com base no terreno
    # Ajuste conforme a escala real do dataset (valores em metros)
    base_pos = np.array([-18, 16, -22])  # do Config.json
    offset_terrain = moved_idx * 0.5     # cada terreno desloca 0.5m
    pos = base_pos + np.array([offset_terrain, 0, 0])

    # Orientação do sonar (apontando para a origem)
    # Rotação em torno de X e Y (graus) – converte para radianos
    rot = Rot.from_euler('xy', [np.radians(rot_x), np.radians(rot_y)], degrees=False)
    rot_matrix = rot.as_matrix()

    # Monta matriz 4x4 (câmera para mundo)
    pose = np.eye(4)
    pose[:3, :3] = rot_matrix
    pose[:3, 3] = pos
    return pose.astype(np.float32)

def load_sonarcloud(data_dir, max_images=5000):
    print(f"Procurando dados em: {data_dir}")

    # Carrega calibração do Config.json
    config_path = os.path.join(data_dir, 'Config.json')
    if os.path.exists(config_path):
        with open(config_path, 'r') as f:
            config = json.load(f)
        for agent in config.get('agents', []):
            for sensor in agent.get('sensors', []):
                if sensor.get('sensor_type') == 'ImagingSonar':
                    cfg = sensor.get('configuration', {})
                    H = cfg.get('RangeBins', 256)
                    W = cfg.get('AzimuthBins', 96)
                    vfov = cfg.get('Elevation', 12.0)
                    hfov = cfg.get('Azimuth', 60.0)
                    min_range = cfg.get('RangeMin', 0.01)
                    max_range = cfg.get('RangeMax', 3.3)
                    break
        print(f"Calibração: H={H}, W={W}, vfov={vfov}, hfov={hfov}, range=[{min_range}, {max_range}]")
    else:
        H, W = 256, 96
        vfov, hfov = 12.0, 60.0
        min_range, max_range = 0.01, 3.3

    # Coleta imagens e poses
    image_files = []
    pose_files = []
    for root, dirs, files in os.walk(data_dir):
        for f in files:
            if f.lower().endswith(('.jpg', '.jpeg', '.png', '.npy')):
                if 'sonar_c' in root:
                    img_path = os.path.join(root, f)
                    image_files.append(img_path)

    # Limita para teste
    if len(image_files) > max_images:
        image_files = image_files[:max_images]
    print(f"Carregando {len(image_files)} imagens...")

    images = []
    poses = []
    for img_path in tqdm(image_files):
        # Imagem
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue
        if img.dtype == np.uint8:
            img = img.astype(np.float32) / 255.0
        if img.ndim == 3:
            img = img.mean(axis=-1)
        if img.shape[0] != H or img.shape[1] != W:
            img = cv2.resize(img, (W, H))
        images.append(torch.from_numpy(img).float())

        # Pose extraída da estrutura de pastas
        pose = parse_pose_from_path(img_path)
        if pose is not None:
            poses.append(torch.from_numpy(pose).float())
        else:
            # Fallback: pose sintética
            angle = 2 * np.pi * len(poses) / len(image_files)
            radius = 2.0
            height = 0.5
            pose = np.eye(4)
            pose[:3, 3] = [radius * np.cos(angle), radius * np.sin(angle), height]
            poses.append(torch.from_numpy(pose).float())

    print(f"Carregadas {len(images)} imagens e {len(poses)} poses.")
    return {
        'sonar_images': images,
        'sensor_poses': poses,
        'min_range': min_range,
        'max_range': max_range,
        'vfov': vfov,
        'hfov': hfov,
        'H': H,
        'W': W,
    }