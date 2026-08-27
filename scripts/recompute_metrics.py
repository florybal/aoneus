"""
Re-avaliar todas as meshes do experimento usando o alinhamento de centróides.

Isso regenera os arquivos em metrics/*.json e metrics.csv com métricas corretas.
"""
import os
import sys
import json
import csv
import numpy as np
import trimesh
from scipy.spatial import cKDTree
from datetime import datetime, timezone

# Adiciona o diretório raiz (pai de scripts/) ao sys.path para importações
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, ROOT_DIR)

from load_sonarcloud import build_gt_pointcloud

EXP_DIR = "experiments/sphere_data_no_terrain_moved_terrain0/0"
GT_DIR = "data/SonarCloud/sphere_data/no_terrain/moved_terrain0/orientation_1"
MESHES_DIR = os.path.join(EXP_DIR, "meshes")
METRICS_DIR = os.path.join(EXP_DIR, "metrics")
CSV_PATH = os.path.join(EXP_DIR, "metrics.csv")
THRESHOLDS = [0.01, 0.02, 0.05, 0.10, 0.20]
ALIGN = "centroid"
N_SAMPLES = 100000

# Limpa o CSV antigo
if os.path.exists(CSV_PATH):
    os.remove(CSV_PATH)

# Carrega GT uma única vez
print("Carregando GT...")
gt_pts = build_gt_pointcloud(GT_DIR)
print(f"GT points: {gt_pts.shape}")

# Processa cada mesh
mesh_files = sorted([f for f in os.listdir(MESHES_DIR) if f.endswith(".obj")])
print(f"Encontradas {len(mesh_files)} meshes.")

# Cabeçalhos CSV
fieldnames = [
    "iteration",
    "mesh",
    "vertices",
    "faces",
    "is_watertight",
    "alignment",
    "accuracy",
    "completeness",
    "chamfer_l1",
    "chamfer_l2",
]
for t in THRESHOLDS:
    fieldnames.append(f"precision_{t:.2f}")
    fieldnames.append(f"recall_{t:.2f}")
    fieldnames.append(f"f1_{t:.2f}")

with open(CSV_PATH, "a", newline="") as csvf:
    writer = csv.DictWriter(csvf, fieldnames=fieldnames)
    writer.writeheader()

for fname in mesh_files:
    iteration = int(fname.replace(".obj", ""))
    mesh_path = os.path.join(MESHES_DIR, fname)

    print(f"\n--- Iteration {iteration} ---")
    mesh = trimesh.load(mesh_path, process=False)

    if len(mesh.vertices) == 0:
        print(f"  Vazio. Skip.")
        continue

    # Amostra pontos da malha predita
    n_pred_samples = min(N_SAMPLES, max(1000, len(mesh.vertices)))
    pred_points, _ = trimesh.sample.sample_surface(mesh, n_pred_samples)

    # Amostra pontos GT
    if len(gt_pts) > N_SAMPLES:
        idx = np.random.choice(len(gt_pts), N_SAMPLES, replace=False)
        gt_eval = gt_pts[idx]
    else:
        gt_eval = gt_pts

    # Alinhamento de centróides
    if ALIGN == "centroid":
        pred_off = np.mean(pred_points, axis=0)
        gt_off = np.mean(gt_eval, axis=0)
        pred_eval = pred_points - pred_off
        gt_eval_aligned = gt_eval - gt_off

    # KDTree e métricas
    gt_tree = cKDTree(gt_eval_aligned)
    pred_tree = cKDTree(pred_eval)
    d_p2g, _ = gt_tree.query(pred_eval, k=1)
    d_g2p, _ = pred_tree.query(gt_eval_aligned, k=1)

    acc = float(np.mean(d_p2g))
    comp = float(np.mean(d_g2p))
    chamfer_l1 = float(acc + comp)
    chamfer_l2 = float(np.mean(d_p2g**2) + np.mean(d_g2p**2))

    print(f"  Accuracy: {acc:.4f}, Completeness: {comp:.4f}, Chamfer L1: {chamfer_l1:.4f}")

    # Monta métricas
    metrics = {
        "iteration": iteration,
        "mesh": os.path.relpath(mesh_path, EXP_DIR),
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "mesh_stats": {
            "vertices": int(len(mesh.vertices)),
            "faces": int(len(mesh.faces)),
            "is_watertight": bool(mesh.is_watertight),
            "is_winding_consistent": bool(mesh.is_winding_consistent),
            "bounds_min": mesh.bounds[0].astype(float).tolist(),
            "bounds_max": mesh.bounds[1].astype(float).tolist(),
            "pred_center": pred_off.astype(float).tolist(),
            "gt_center": gt_off.astype(float).tolist(),
        },
        "geometry": {
            "alignment": ALIGN,
            "accuracy": acc,
            "completeness": comp,
            "chamfer_l1": chamfer_l1,
            "chamfer_l2": chamfer_l2,
            "gt_points_count": int(len(gt_eval)),
            "pred_points_count": int(len(pred_points)),
        },
        "thresholds": {},
    }

    row = {
        "iteration": iteration,
        "mesh": metrics["mesh"],
        "vertices": metrics["mesh_stats"]["vertices"],
        "faces": metrics["mesh_stats"]["faces"],
        "is_watertight": metrics["mesh_stats"]["is_watertight"],
        "alignment": ALIGN,
        "accuracy": acc,
        "completeness": comp,
        "chamfer_l1": chamfer_l1,
        "chamfer_l2": chamfer_l2,
    }

    for t in THRESHOLDS:
        prec = float(np.mean(d_p2g < t))
        rec = float(np.mean(d_g2p < t))
        f1 = float(2.0 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0
        metrics["thresholds"][f"{t:.6f}"] = {
            "precision": prec,
            "recall": rec,
            "f1": f1,
        }
        row[f"precision_{t:.2f}"] = prec
        row[f"recall_{t:.2f}"] = rec
        row[f"f1_{t:.2f}"] = f1
        print(f"  Threshold {t:.2f}m: P={prec*100:.2f}%, R={rec*100:.2f}%, F1={f1*100:.2f}%")

    # Salva JSON
    json_path = os.path.join(METRICS_DIR, f"{iteration:08d}.json")
    with open(json_path, "w") as f:
        json.dump(metrics, f, indent=2)

    # Escreve no CSV
    with open(CSV_PATH, "a", newline="") as csvf:
        writer = csv.DictWriter(csvf, fieldnames=fieldnames)
        writer.writerow(row)

print("\n=== FIM ===")
print(f"Resultados salvos em {METRICS_DIR}/ e {CSV_PATH}")