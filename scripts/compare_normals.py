#!/usr/bin/env python3
"""
Gera uma imagem composta (lado-a-lado) juntando:
  - validations_fine/<iter>_0_<idx>.png   (NeuS render + GT lado-a-lado)
  - normals/<iter>_0_<idx>.png            (mesh render via pyrender)

Saída em <exp_dir>/comparisons/<iter>_0_<idx>.png

Se validations_fine/ estiver vazio, gera apenas o mesh render.
"""
import argparse
import os
import sys
from pathlib import Path

import cv2
import numpy as np


def compose(validation_path: str | None, mesh_path: str, out_path: str,
            target_height: int = 540) -> bool:
    mesh = cv2.imread(mesh_path, cv2.IMREAD_COLOR)
    if mesh is None:
        return False
    h, w = mesh.shape[:2]
    scale = target_height / float(h)
    mesh_resized = cv2.resize(mesh, (int(w * scale), target_height),
                              interpolation=cv2.INTER_AREA)

    panels = []
    labels = []
    if validation_path and os.path.exists(validation_path):
        val = cv2.imread(validation_path, cv2.IMREAD_COLOR)
        if val is not None:
            vh, vw = val.shape[:2]
            vscale = target_height / float(vh)
            val_resized = cv2.resize(val, (int(vw * vscale), target_height),
                                     interpolation=cv2.INTER_AREA)
            panels.append(val_resized)
            labels.append("validation_fine (GT+render)")

    panels.append(mesh_resized)
    labels.append("mesh render (normals)")

    sep = np.full((target_height, 8, 3), 32, dtype=np.uint8)
    out = panels[0]
    for p in panels[1:]:
        out = np.concatenate([out, sep, p], axis=1)

    # Label strip no topo
    label_h = 30
    lbl_strip = np.full((label_h, out.shape[1], 3), 16, dtype=np.uint8)
    out = np.concatenate([lbl_strip, out], axis=0)
    x = 10
    for lbl in labels:
        cv2.putText(out, lbl, (x, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                    (255, 255, 255), 1, cv2.LINE_AA)
        x += int(len(lbl) * 11) + 30

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    cv2.imwrite(out_path, out)
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--experiments_root", default="experiments")
    ap.add_argument("--experiments", nargs="*", default=None)
    ap.add_argument("--target_height", type=int, default=540)
    args = ap.parse_args()

    root = Path(args.experiments_root)
    exp_dirs = []
    for d in sorted(root.iterdir()):
        if not d.is_dir():
            continue
        if args.experiments and d.name not in args.experiments:
            continue
        # Pode ter /0 ou /<timestamp>
        for sub in sorted(d.iterdir()):
            if not sub.is_dir():
                continue
            if (sub / "normals").is_dir():
                exp_dirs.append((d.name, sub))

    total = 0
    for exp_name, run_dir in exp_dirs:
        normals_dir = run_dir / "normals"
        valid_dir = run_dir / "validations_fine"
        out_dir = run_dir / "comparisons"
        pngs = sorted(normals_dir.glob("*.png"))
        if not pngs:
            print(f"[skip] {exp_name}/{run_dir.name}: normals/ vazio.")
            continue
        n_made = 0
        n_with_val = 0
        for png in pngs:
            val_path = valid_dir / png.name if valid_dir.exists() else None
            out_path = out_dir / png.name
            ok = compose(str(val_path) if val_path else None,
                         str(png), str(out_path), args.target_height)
            if ok:
                n_made += 1
                if val_path and val_path.exists():
                    n_with_val += 1
        print(f"[{exp_name}] {n_made} comparacoes geradas em {out_dir.name}/"
              f"  ({n_with_val} com validation_fine)")
        total += n_made

    print(f"\n[OK] {total} comparacoes geradas em comparisons/.")


if __name__ == "__main__":
    main()