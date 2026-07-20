import os
import glob
import torch
import numpy as np
import cv2
import json
from tqdm import tqdm
from scipy.spatial.transform import Rotation as Rot

def load_sonarcloud(data_dir, max_images=200):
    """
    Carrega o dataset SonarCloud recursivamente.
    Como não há arquivos de pose, gera uma trajetória sintética.
    """
    print(f"Procurando dados em: {data_dir}")

    # ============================================================
    # 1. Parâmetros de calibração (do Config.json)
    # ============================================================
    config_path = os.path.join(data_dir, 'Config.json')
    if os.path.exists(config_path):
        with open(config_path, 'r') as f:
            config = json.load(f)
        # Extrai parâmetros do sonar
        for agent in config.get('agents', []):
            for sensor in agent.get('sensors', []):
                if sensor.get('sensor_type') == 'ImagingSonar':
                    cfg = sensor.get('configuration', {})
                    H = cfg.get('RangeBins', 256)
                    W = cfg.get('AzimuthBins', 96)
                    vfov = cfg.get('Elevation', 12.0)      # graus
                    hfov = cfg.get('Azimuth', 60.0)        # graus
                    min_range = cfg.get('RangeMin', 0.01)
                    max_range = cfg.get('RangeMax', 3.3)
                    break
        print(f"Calibração carregada: H={H}, W={W}, vfov={vfov}, hfov={hfov}, range=[{min_range}, {max_range}]")
    else:
        H, W = 256, 96
        vfov, hfov = 12.0, 60.0
        min_range, max_range = 0.01, 3.3
        print("Config.json não encontrado. Usando valores padrão do sonar.")

    # ============================================================
    # 2. Coletar arquivos de imagem (sonar_c/*.jpg)
    # ============================================================
    image_files = []
    for root, dirs, files in os.walk(data_dir):
        for f in files:
            if f.lower().endswith(('.jpg', '.jpeg', '.png', '.npy')):
                # Só pega imagens que estão em pastas 'sonar_c'
                if 'sonar_c' in root:
                    image_files.append(os.path.join(root, f))

    # Limita o número de imagens para teste
    if len(image_files) > max_images:
        image_files = image_files[:max_images]
        print(f"Limitando a {max_images} imagens para teste.")

    print(f"Encontradas {len(image_files)} imagens.")

    # ============================================================
    # 3. Carregar imagens
    # ============================================================
    images = []
    for img_path in tqdm(image_files, desc="Carregando imagens"):
        if img_path.endswith('.npy'):
            img = np.load(img_path)
        else:
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                # Tenta ler com outro método
                img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
                if img is None:
                    continue
        # Normalização
        if img.dtype == np.uint8:
            img = img.astype(np.float32) / 255.0
        if img.ndim == 3:
            img = img.mean(axis=-1)
        # Redimensionar para HxW (padrão do sonar)
        if img.shape[0] != H or img.shape[1] != W:
            img = cv2.resize(img, (W, H))
        images.append(torch.from_numpy(img).float())

    print(f"Carregadas {len(images)} imagens válidas.")

    if len(images) == 0:
        raise RuntimeError("Nenhuma imagem válida foi carregada. Verifique o caminho e os arquivos.")

    # ============================================================
    # 4. Gerar poses sintéticas (órbita circular)
    # ============================================================
    num_images = len(images)
    radius = 2.0      # raio da órbita em metros
    height = 0.5      # altura do sonar
    poses = []
    for i in range(num_images):
        angle = 2 * np.pi * i / num_images
        x = radius * np.cos(angle)
        y = radius * np.sin(angle)
        z = height
        # Pose olhando para a origem
        # Matriz 4x4: rotação + translação
        eye = np.array([x, y, z])
        center = np.array([0, 0, 0])
        up = np.array([0, 0, 1])
        # Criar matriz de visualização (câmera para mundo)
        # Para simplificar, usamos uma matriz identidade com translação
        pose = np.eye(4)
        pose[:3, 3] = eye
        # Opcional: adicionar rotação para olhar para o centro
        # (mais realista, mas não necessário para teste)
        poses.append(torch.from_numpy(pose.astype(np.float32)).float())

    print(f"Geradas {len(poses)} poses sintéticas.")

    # ============================================================
    # 5. Retornar dicionário
    # ============================================================
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