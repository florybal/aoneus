import torch
import numpy as np
from models.fields import SDFNetwork
from models.renderer import extract_geometry
from pyhocon import ConfigFactory
import trimesh

conf = ConfigFactory.parse_file("confs/sonar_cloud/sonar_cloud_sonar.conf")
sdf_network = SDFNetwork(**conf['model.sdf_network']).cuda()
checkpoint = torch.load("experiments/SonarCloud/1706110819/checkpoints/ckpt_00005200.pth", map_location="cuda")
sdf_network.load_state_dict(checkpoint['sdf_network_fine'])

bound_min = torch.tensor([-1.5, -1.5, -1.5], device="cuda")
bound_max = torch.tensor([1.5, 1.5, 1.5], device="cuda")
resolution = 128

for th in [0.0, 0.01, 0.02, 0.05, -0.01, -0.02, -0.05]:
    vertices, triangles = extract_geometry(bound_min, bound_max, resolution, th,
                                           query_func=lambda pts: -sdf_network.sdf(pts))
    if len(vertices) > 0:
        mesh = trimesh.Trimesh(vertices, triangles)
        mesh.export(f"mesh_th{th}.obj")
        print(f"✅ Threshold {th}: {len(vertices)} vértices, {len(triangles)} faces")
    else:
        print(f"❌ Threshold {th}: vazio")