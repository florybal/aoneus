#!/usr/bin/env python3
"""
Renderiza múltiplas vistas de cada malha em todos os experimentos,
salvando em <exp_dir>/normals/ no padrão {iter:08d}_0_{view:02d}.png.

Inspirado em visualize_mesh.py (pyrender + trimesh).
Para casar com o exemplo reduzido do aoneus
(experiments/reduced_baseline_0.6x_joint/1706110819/normals/), que usa
o mesmo padrão de nomes gerado pelo NeuS/exp_runner.py:
    '{:0>8d}_{}_{}.png'.format(self.iter_step, i, idx)

Uso:
    python scripts/batch_visualize.py
    python scripts/batch_visualize.py --num_views 8 --img_size 512x512
    python scripts/batch_visualize.py --experiments 500lbs_data_more_ruggedy_terrain_moved_terrain0
"""
import argparse
import os
import sys
from pathlib import Path

import numpy as np
import trimesh
import pyrender
import cv2


def parse_size(s: str):
    w, h = s.lower().split("x")
    return int(w), int(h)


def render_views(mesh_path: str, out_dir: str, iter_id: int, num_views: int,
                 width: int, height: int, light_intensity: float = 3.0) -> int:
    """Renderiza `num_views` vistas orbitais da malha e salva em out_dir.

    Retorna o número de imagens geradas.
    """
    os.makedirs(out_dir, exist_ok=True)

    mesh = trimesh.load(mesh_path, force="mesh")
    if mesh.is_empty:
        print(f"  [skip] malha vazia: {mesh_path}")
        return 0

    # Centraliza a malha na origem para visualização estável.
    mesh.apply_translation(-mesh.centroid)

    # Frame pyrender (rotacionado 180° em X para alinhar com sistema do NeuS).
    rotation_x = trimesh.transformations.rotation_matrix(np.pi, [1, 0, 0])
    mesh.apply_transform(rotation_x)

    scene = pyrender.Scene(bg_color=[0, 0, 0, 0])
    scene.add(pyrender.Mesh.from_trimesh(mesh))

    # Câmera perspectiva a 3.0m de distância, yfov π/3 (igual visualize_mesh.py).
    camera = pyrender.PerspectiveCamera(yfov=np.pi / 3.0)
    renderer = pyrender.OffscreenRenderer(viewport_width=width, viewport_height=height)

    n = 0
    for v in range(num_views):
        # Orbita no plano horizontal ao redor da malha.
        angle = 2.0 * np.pi * v / num_views
        cam_pos = np.array([3.0 * np.sin(angle), 0.0, 3.0 * np.cos(angle)])
        cam_pose = np.eye(4)
        cam_pose[:3, 3] = cam_pos
        # Câmera olha para a origem.
        forward = -cam_pos / np.linalg.norm(cam_pos)
        up = np.array([0.0, 1.0, 0.0])
        right = np.cross(forward, up)
        if np.linalg.norm(right) < 1e-6:
            # Caso a câmera esteja no eixo Y
            up = np.array([0.0, 0.0, 1.0])
            right = np.cross(forward, up)
        right /= np.linalg.norm(right)
        up = np.cross(right, forward)
        
        # Pyrender usa: col0=X(right), col1=Y(up), col2=Z(-forward)
        cam_pose[:3, 0] = right
        cam_pose[:3, 1] = up
        cam_pose[:3, 2] = -forward

        # Luz direcional próxima da câmera.
        light = pyrender.DirectionalLight(color=[1.0, 1.0, 1.0],
                                          intensity=light_intensity)
        cam_node = scene.add(camera, pose=cam_pose)
        light_node = scene.add(light, pose=cam_pose)

        color, _ = renderer.render(scene)
        scene.remove_node(cam_node)
        scene.remove_node(light_node)

        # Nome compatível com o padrão NeuS: {iter:08d}_{0}_{view:02d}.png
        out_path = os.path.join(out_dir, f"{iter_id:08d}_0_{v:02d}.png")
        cv2.imwrite(out_path, color[:, :, ::-1])  # RGB -> BGR
        n += 1

    renderer.delete()
    return n


def find_mesh_dirs(experiments_root: str, only: list[str] | None = None):
    """Retorna lista de (exp_dir, meshes_dir, normals_dir) a processar."""
    results = []
    exp_root = Path(experiments_root)
    for exp_dir in sorted(exp_root.iterdir()):
        if not exp_dir.is_dir():
            continue
        if only and exp_dir.name not in only:
            continue

        # Pode ter subpasta timestampada (ex: reduced_baseline_0.6x_joint/1706110819)
        # ou diretamente /0 (ex: 500lbs_..._moved_terrain0/0).
        candidates = []
        for sub in sorted(exp_dir.iterdir()):
            if not sub.is_dir():
                continue
            if (sub / "meshes").is_dir():
                candidates.append(sub)
        if candidates:
            for c in candidates:
                results.append((exp_dir.name, c, c / "meshes", c / "normals"))
    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--experiments_root", default="experiments")
    ap.add_argument("--experiments", nargs="*", default=None,
                    help="Restringe a um subconjunto de nomes de experimento.")
    ap.add_argument("--num_views", type=int, default=8)
    ap.add_argument("--img_size", type=parse_size, default=(512, 512),
                    help="WxH, default 512x512.")
    ap.add_argument("--light", type=float, default=3.0)
    args = ap.parse_args()

    width, height = args.img_size
    entries = find_mesh_dirs(args.experiments_root, args.experiments)
    if not entries:
        print(f"[ERRO] Nenhum experimento com pasta meshes/ em {args.experiments_root}",
              file=sys.stderr)
        sys.exit(1)

    grand_total = 0
    for exp_name, run_dir, meshes_dir, normals_dir in entries:
        # Pega todas as malhas (.obj e .ply), ordenadas por número de iteração.
        mesh_paths = sorted(
            list(meshes_dir.glob("*.obj")) + list(meshes_dir.glob("*.ply")),
            key=lambda p: int(p.stem),
        )
        if not mesh_paths:
            print(f"[skip] {exp_name}/{run_dir.name}: nenhuma malha encontrada.")
            continue
        print(f"\n=== {exp_name}/{run_dir.name}: {len(mesh_paths)} malha(s), "
              f"{args.num_views} vistas cada ===")

        exp_total = 0
        for mesh_path in mesh_paths:
            iter_id = int(mesh_path.stem)
            n = render_views(
                str(mesh_path), str(normals_dir),
                iter_id, args.num_views, width, height, args.light,
            )
            print(f"  iter={iter_id:08d} -> {n} imagens em {normals_dir.name}/")
            exp_total += n

        print(f"  [{exp_name}] total: {exp_total} imagens")
        grand_total += exp_total

    print(f"\n[OK] {grand_total} imagens geradas em normals/.")


if __name__ == "__main__":
    main()