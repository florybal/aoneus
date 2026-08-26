import os
import torch
import numpy as np
import cv2
import json
from tqdm import tqdm
from scipy.spatial.transform import Rotation as Rot
import math

from scipy.spatial.transform import Rotation as Rot
import numpy as np
import os
from data.sonarcloud.reconstruct_data import reconstruct3d

  
def build_gt_pointcloud(self):
        """
        Constrói a nuvem GT a partir dos 8 depth maps + poses.

        Estrutura esperada:

        .../orientation_1/
            depth/
                depth_0.tiff
                ...
                depth_7.tiff
            rgb/
                1.jpg
                ...
                8.jpg
            poses_cluster/
                pose_1.csv
                ...
                pose_8.csv
        """

        # ---------------------------------------------------------
        # IMPORTANTE:
        # descobrir a pasta da orientação atual
        # ---------------------------------------------------------

        # Ajuste esta variável conforme seu dataset.
        gt_root = self.conf.get_string(
            "conf.gt_root",
            default=None
        )

        if gt_root is None:
            raise RuntimeError(
                "conf.gt_root não foi configurado."
            )

        print("GT root:", gt_root)

        all_points = []

        # ---------------------------------------------------------
        # câmera
        # ---------------------------------------------------------

        hfov_degrees = 60.0
        vfov_degrees = 60.0

        hFov = math.radians(hfov_degrees)
        vFov = math.radians(vfov_degrees)

        # ---------------------------------------------------------
        # 8 orientações / depth maps
        # ---------------------------------------------------------

        for i in range(1, 9):

            depth_path = os.path.join(
                gt_root,
                "depth",
                f"depth_{i-1}.tiff"
            )

            pose_path = os.path.join(
                gt_root,
                "poses_cluster",
                f"pose_{i}.csv"
            )

            rgb_path = os.path.join(
                gt_root,
                "rgb",
                f"{i}.jpg"
            )

            if not os.path.exists(depth_path):
                raise FileNotFoundError(depth_path)

            if not os.path.exists(pose_path):
                raise FileNotFoundError(pose_path)

            # -----------------------------------------------------
            # depth
            # -----------------------------------------------------

            depthmap = cv2.imread(
                depth_path,
                cv2.IMREAD_ANYDEPTH
            )

            if depthmap is None:
                raise RuntimeError(
                    f"Não foi possível ler {depth_path}"
                )

            depthmap = depthmap.astype(np.float32)

            # Seu código original usa 10 como inválido
            mask = depthmap != 10.0
            depthmap[~mask] = 0.0

            # -----------------------------------------------------
            # RGB
            # -----------------------------------------------------

            if os.path.exists(rgb_path):
                image = cv2.imread(rgb_path)

                if image is None:
                    raise RuntimeError(
                        f"Não foi possível ler {rgb_path}"
                    )

                image = cv2.cvtColor(
                    image,
                    cv2.COLOR_BGR2RGB
                )
            else:
                # Não precisamos realmente da cor para as métricas.
                image = np.zeros(
                    (depthmap.shape[0], depthmap.shape[1], 3),
                    dtype=np.uint8
                )

            h, w = image.shape[:2]

            # -----------------------------------------------------
            # redimensiona depth se necessário
            # -----------------------------------------------------

            if depthmap.shape[0] != h or depthmap.shape[1] != w:

                depthmap = cv2.resize(
                    depthmap,
                    (w, h),
                    interpolation=cv2.INTER_NEAREST
                )

            # -----------------------------------------------------
            # intrínseca
            # -----------------------------------------------------

            cx = w / 2.0
            cy = h / 2.0

            fx = w / (2.0 * math.tan(hFov / 2.0))
            fy = h / (2.0 * math.tan(vFov / 2.0))

            camera_params = np.array([
                [fx, 0, cx],
                [0, fy, cy],
                [0, 0, 1]
            ], dtype=np.float32)

            # -----------------------------------------------------
            # pose
            # -----------------------------------------------------

            data = np.loadtxt(
                pose_path,
                delimiter=","
            ).reshape(-1)

            x = data[2]
            y = data[1]
            z = data[0]
            yaw = data[5]

            # -----------------------------------------------------
            # reconstrução
            # -----------------------------------------------------

            _, transformed_points = reconstruct.reconstruct3d(
                image,
                depthmap,
                x,
                y,
                z,
                yaw,
                camera_params,
                step=1,
                mesh=False
            )

            pts = np.asarray(
                transformed_points,
                dtype=np.float32
            )

            # remove pontos inválidos
            valid = np.isfinite(pts).all(axis=1)
            pts = pts[valid]

            if len(pts) > 0:
                all_points.append(pts)

            print(
                f"GT depth {i}: {len(pts)} pontos"
            )

        if not all_points:
            raise RuntimeError(
                "Nenhum ponto GT foi gerado."
            )

        gt_points = np.concatenate(
            all_points,
            axis=0
        )

        print(
            "GT point cloud:",
            gt_points.shape
        )

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

def load_sonarcloud(data_dir, max_images=None):

    print(f"\n{'=' * 70}")
    print(f"SONARCLOUD DATASET")
    print(f"Root: {data_dir}")
    print(f"{'=' * 70}")

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

        images.append(
            torch.from_numpy(img).float()
        )

        poses.append(
            torch.from_numpy(pose).float()
        )

    print()
    print("==========================================")
    print("CHECK DATASET")
    print("==========================================")
    print(f"Dataset: {data_dir}")
    print(f"Imagens: {len(images)}")
    print(f"Poses:   {len(poses)}")

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

        "metadata": metadata
    }