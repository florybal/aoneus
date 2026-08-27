import os
import sys
import json
import math
import cv2
import numpy as np
try:
    import torch
except ImportError:
    torch = None
try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable, *args, **kwargs):
        return iterable
from scipy.spatial.transform import Rotation as Rot
from scipy.spatial import cKDTree
from PIL import Image

try:
    from data.SonarCloud.reconstruct_data import reconstruct3d, reconstruct3d_points
except (ImportError, ModuleNotFoundError, ValueError):
    try:
        from reconstruct_data import reconstruct3d, reconstruct3d_points
    except (ImportError, ModuleNotFoundError, ValueError):
        import importlib.util
        reconstruct3d = None
        reconstruct3d_points = None
        for p in [
            os.path.join(os.path.dirname(__file__), "data", "SonarCloud", "reconstruct_data.py"),
            os.path.join(os.path.dirname(__file__), "reconstruct_data.py"),
            os.path.abspath("data/SonarCloud/reconstruct_data.py")
        ]:
            if os.path.exists(p):
                spec = importlib.util.spec_from_file_location("reconstruct_data", p)
                mod = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(mod)
                reconstruct3d = getattr(mod, "reconstruct3d", None)
                reconstruct3d_points = getattr(mod, "reconstruct3d_points", None)
                break


def find_sonarcloud_root(path):
    """
    Localiza o diretório raiz do dataset SonarCloud contendo poses_cluster e rgb.
    """
    p = os.path.abspath(path)
    while p != os.path.dirname(p):
        if os.path.basename(p).lower() == "sonarcloud":
            return p
        candidate = os.path.join(p, "data", "SonarCloud")
        if os.path.exists(candidate) and os.path.isdir(candidate):
            return candidate
        p = os.path.dirname(p)
    
    local_path = os.path.join(os.getcwd(), "data", "SonarCloud")
    if os.path.exists(local_path):
        return local_path
    return None


def build_gt_pointcloud(
    data_dir,
    sonar_cloud_root=None,
    apply_rotation=True,
    filter_outliers=True,
    nb_neighbors=10,
    std_ratio=2.0
):
    """
    Constrói a nuvem de pontos Ground Truth (GT) 3D a partir dos depth maps do SonarCloud
    utilizando a reprojeção 3D de reconstruct_data.py.

    Args:
        data_dir (str): Caminho para a cena ou orientação (ex: data/SonarCloud/sphere_data/.../orientation_1)
        sonar_cloud_root (str, optional): Raiz do SonarCloud contendo poses_cluster/ e rgb/
        apply_rotation (bool): Aplica rotação de -90 graus no eixo X (alinhamento padrão do SonarCloud)
        filter_outliers (bool): Aplica remoção estatística de outliers
        nb_neighbors (int): Número de vizinhos para filtro de outliers
        std_ratio (float): Razão de desvio padrão para corte de outliers

    Returns:
        np.ndarray: Nuvem de pontos GT (N, 3) float32 ou None se nenhum depth for encontrado
    """
    if sonar_cloud_root is None:
        sonar_cloud_root = find_sonarcloud_root(data_dir)

    if sonar_cloud_root is None:
        print(f"[build_gt_pointcloud] Aviso: Não foi possível determinar a raiz do SonarCloud para: {data_dir}")
        return None

    # Localizar pastas 'depth'
    depth_dirs = []
    if os.path.basename(os.path.normpath(data_dir)) == "depth":
        depth_dirs.append(data_dir)
    else:
        for root, dirs, files in os.walk(data_dir):
            if os.path.basename(root) == "depth":
                depth_dirs.append(root)

    depth_dirs.sort()
    if not depth_dirs:
        print(f"[build_gt_pointcloud] Nenhum diretório 'depth' encontrado em {data_dir}")
        return None

    all_valid_points = []
    hfov_rad = math.radians(60.0)
    vfov_rad = math.radians(60.0)

    for d_dir in depth_dirs:
        # Verifica se é dataset UXO
        is_uxo = any(k in d_dir.lower() for k in ["uxo", "500lbs", "mortar", "mine"])
        poses_dir_name = "poses_uxos_cluster" if is_uxo else "poses_cluster"
        poses_dir = os.path.join(sonar_cloud_root, poses_dir_name)
        rgb_dir = os.path.join(sonar_cloud_root, "rgb")

        for i in range(1, 9):
            depth_path = os.path.join(d_dir, f"depth_{i-1}.tiff")
            pose_path = os.path.join(poses_dir, f"pose_{i}.csv")
            rgb_path = os.path.join(rgb_dir, f"{i}.jpg")

            if not os.path.exists(depth_path) or not os.path.exists(pose_path):
                continue

            # Ler depth
            depth_img = Image.open(depth_path)
            depthmap = np.array(depth_img, dtype=np.float32)
            mask = (depthmap != 10.0) & (depthmap > 0.0) & np.isfinite(depthmap)
            depthmap[~mask] = 0.0

            h, w = depthmap.shape[:2]
            cx, cy = w / 2.0, h / 2.0
            fx = w / (2.0 * math.tan(hfov_rad / 2.0))
            fy = h / (2.0 * math.tan(vfov_rad / 2.0))
            camera_params = np.array([
                [fx, 0, cx],
                [0, fy, cy],
                [0, 0, 1]
            ], dtype=np.float32)

            # Ler pose
            data = np.loadtxt(pose_path, delimiter=",").reshape(-1)
            x = float(data[2])
            y = float(data[1])
            z = float(data[0])
            yaw = float(data[5])
            if is_uxo:
                yaw = yaw - 1.57

            if reconstruct3d_points is not None:
                pts = reconstruct3d_points(
                    depthmap, x, y, z, yaw, camera_params, valid_only=True
                )
            else:
                if os.path.exists(rgb_path):
                    image = cv2.imread(rgb_path)
                    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                else:
                    image = np.zeros((h, w, 3), dtype=np.uint8)

                if image.shape[:2] != (h, w):
                    image = cv2.resize(image, (w, h))

                _, transformed_points = reconstruct3d(
                    image, depthmap, x, y, z, yaw, camera_params, step=1, mesh=False
                )
                pts = np.asarray(transformed_points, dtype=np.float32)
                valid = (depthmap > 0.0).reshape(-1)
                pts = pts[valid]

            if len(pts) > 0:
                all_valid_points.append(pts)

    if not all_valid_points:
        print(f"[build_gt_pointcloud] Nenhum ponto válido foi reconstruído para {data_dir}")
        return None

    gt_points = np.concatenate(all_valid_points, axis=0)

    # Rotação de alinhamento em torno do eixo X (-90 graus)
    if apply_rotation:
        theta = np.radians(-90.0)
        rot_x = np.array([
            [1.0, 0.0, 0.0],
            [0.0, np.cos(theta), -np.sin(theta)],
            [0.0, np.sin(theta), np.cos(theta)]
        ], dtype=np.float32)
        gt_points = np.dot(gt_points, rot_x.T)

    # Filtragem estatística de outliers
    if filter_outliers and len(gt_points) > nb_neighbors:
        tree = cKDTree(gt_points)
        dists, _ = tree.query(gt_points, k=nb_neighbors + 1)
        mean_d = np.mean(dists[:, 1:], axis=1)
        thresh = np.mean(mean_d) + std_ratio * np.std(mean_d)
        gt_points = gt_points[mean_d <= thresh]

    gt_points = gt_points.astype(np.float32)
    print(f"[build_gt_pointcloud] GT point cloud gerada: {gt_points.shape} pontos")
    return gt_points

def parse_pose_from_path(path):
    """
    Extrai moved_terrain e orientation do caminho.

    Exemplo:
    .../boat_data/more_ruggedy_terrain/moved_terrain1/orientation_5/sonar_c/FLSc_10.jpg
    """

    parts = path.split(os.sep)

    moved_idx = None
    orientation_idx = None

    for p in parts:
        if p.startswith("moved_terrain"):
            moved_idx = int(
                p.replace("moved_terrain", "")
            )

        elif p.startswith("orientation_"):
            orientation_idx = int(
                p.replace("orientation_", "")
            )

    if moved_idx is None or orientation_idx is None:
        return None

    # ==================================================
    # POSIÇÃO
    # ==================================================

    base_pos = np.array(
        [-18.0, 16.0, -22.0],
        dtype=np.float32
    )

    position = base_pos.copy()

    # ATENÇÃO:
    # Isto ainda é uma hipótese.
    position[0] += moved_idx * 0.5

    # ==================================================
    # ORIENTAÇÃO
    # ==================================================

    orientation_angles = {
        1: 0.0,
        2: 45.0,
        3: 90.0,
        4: 135.0,
        5: 180.0,
        6: 225.0,
        7: 270.0,
        8: 315.0,
    }

    if orientation_idx not in orientation_angles:
        raise ValueError(
            f"Orientação inválida: orientation_{orientation_idx}"
        )

    yaw_deg = orientation_angles[orientation_idx]

    R = Rot.from_euler(
        "z",
        yaw_deg,
        degrees=True
    ).as_matrix()

    # ==================================================
    # MATRIZ HOMOGÊNEA
    # ==================================================

    pose = np.eye(4, dtype=np.float32)

    pose[:3, :3] = R
    pose[:3, 3] = position

    return pose

def load_sonarcloud(data_dir, max_images=None, load_gt=True):

    print(f"\n{'=' * 70}")
    print(f"SONARCLOUD DATASET")
    print(f"Root: {data_dir}")
    print(f"{'=' * 70}")

    # --------------------------------------------------
    # GROUND TRUTH POINT CLOUD (de reconstruct_data.py)
    # --------------------------------------------------
    gt_points = None
    if load_gt:
        try:
            gt_points = build_gt_pointcloud(data_dir)
            if gt_points is not None:
                print(f"[load_sonarcloud] Ground Truth carregado com sucesso: {gt_points.shape[0]} pontos.")
            else:
                print("[load_sonarcloud] Nenhum Ground Truth encontrado para esta pasta.")
        except Exception as e:
            print(f"[load_sonarcloud] Erro ao construir GT point cloud: {e}")

    # --------------------------------------------------
    # CALIBRAÇÃO
    # --------------------------------------------------

    config_path = os.path.join(data_dir, "Config.json")

    if os.path.exists(config_path):

        with open(config_path, "r") as f:
            config = json.load(f)

        H = 256
        W = 96

        vfov = math.radians(12.0)
        hfov = math.radians(60.0)

        min_range = 0.01
        max_range = 3.3

        for agent in config.get("agents", []):

            for sensor in agent.get("sensors", []):

                if sensor.get("sensor_type") == "ImagingSonar":

                    cfg = sensor.get("configuration", {})

                    H = cfg.get("RangeBins", H)
                    W = cfg.get("AzimuthBins", W)

                    vfov = math.radians(
                        cfg.get("Elevation", 12.0)
                    )

                    hfov = math.radians(
                        cfg.get("Azimuth", 60.0)
                    )

                    min_range = cfg.get(
                        "RangeMin",
                        min_range
                    )

                    max_range = cfg.get(
                        "RangeMax",
                        max_range
                    )

                    break

    else:

        H = 256
        W = 96

        vfov = math.radians(12.0)
        hfov = math.radians(60.0)

        min_range = 0.01
        max_range = 3.3

    print(
        f"H={H}, W={W}, "
        f"FOV={math.degrees(hfov):.2f} x "
        f"{math.degrees(vfov):.2f}, "
        f"range=[{min_range}, {max_range}]"
    )

    # --------------------------------------------------
    # ENCONTRAR IMAGENS
    # --------------------------------------------------

    image_files = []

    for root, dirs, files in os.walk(data_dir):

        # Só queremos arquivos dentro de sonar_c
        if os.path.basename(root) != "sonar_c":
            continue

        for f in files:

            if f.lower().endswith(
                (".jpg", ".jpeg", ".png")
            ):
                image_files.append(
                    os.path.join(root, f)
                )

    image_files.sort()

    if max_images is not None:
        image_files = image_files[:max_images]

    print(f"Encontradas {len(image_files)} imagens.")

    # --------------------------------------------------
    # CARREGAR
    # --------------------------------------------------

    images = []
    poses = []
    metadata = []

    for img_path in tqdm(image_files):

        img = cv2.imread(
            img_path,
            cv2.IMREAD_GRAYSCALE
        )

        if img is None:
            print(f"ERRO lendo: {img_path}")
            continue

        # JPG é uint8 → [0, 255]
        # normaliza para [0, 1]
        img = img.astype(np.float32) / 255.0

        print(
            f"Imagem: {os.path.basename(img_path)} | "
            f"shape original: {img.shape}"
        )

        if img.shape != (H, W):
            img = cv2.resize(
                img,
                (W, H),
                interpolation=cv2.INTER_AREA
            )

        pose = parse_pose_from_path(img_path)

        if pose is None:
            print(
                f"Não foi possível determinar pose: {img_path}"
            )
            continue

        if torch is not None:
            images.append(torch.from_numpy(img).float())
            poses.append(torch.from_numpy(pose).float())
        else:
            images.append(img)
            poses.append(pose)

    print()
    print("==========================================")
    print("CHECK DATASET")
    print("==========================================")
    print(f"Dataset: {data_dir}")
    print(f"Imagens: {len(images)}")
    print(f"Poses:   {len(poses)}")
    if gt_points is not None:
        print(f"GT Pts:  {gt_points.shape}")

    for i in range(min(10, len(images))):

        print()
        print(f"[{i}]")
        print(f"shape: {images[i].shape}")
        print(f"min:   {images[i].min().item():.4f}")
        print(f"max:   {images[i].max().item():.4f}")
        print("pose:")
        print(poses[i])

    print("==========================================")

    return {
        "sonar_images": images,
        "sensor_poses": poses,

        "min_range": min_range,
        "max_range": max_range,

        "vfov": vfov,
        "hfov": hfov,

        "H": H,
        "W": W,

        "metadata": metadata,
        "gt_points": gt_points
    }