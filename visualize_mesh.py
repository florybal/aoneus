import pyrender
import trimesh
import argparse
import numpy as np
import os
import cv2

parser = argparse.ArgumentParser()
parser.add_argument("--mesh", required=True, help="Caminho para .ply ou .obj")
parser.add_argument("--output", default="mesh.png", help="Nome da imagem de saída")
args = parser.parse_args()

if not os.path.exists(args.mesh):
    print(f"Erro: arquivo não encontrado: {args.mesh}")
    exit(1)

# Carrega a malha
mesh = trimesh.load(args.mesh)
if mesh.is_empty:
    print("Erro: malha vazia (0 vértices/faces).")
    exit(1)

# Cena
scene = pyrender.Scene()
mesh_pyrender = pyrender.Mesh.from_trimesh(mesh)
scene.add(mesh_pyrender)

# Câmera (posicionada para enquadrar a malha)
camera = pyrender.PerspectiveCamera(yfov=np.pi / 3.0)
camera_pose = np.eye(4)
camera_pose[:3, 3] = [0, 0, 3.0]  # distância da câmera
scene.add(camera, pose=camera_pose)

# Luz
light = pyrender.DirectionalLight(color=[1.0, 1.0, 1.0], intensity=3.0)
scene.add(light, pose=camera_pose)

# Renderizador offscreen
r = pyrender.OffscreenRenderer(viewport_width=1920, viewport_height=1080)
color, depth = r.render(scene)
r.delete()

# Salva a imagem
cv2.imwrite(args.output, color[:, :, ::-1])  # BGR -> RGB
print(f"Imagem salva em: {args.output}")