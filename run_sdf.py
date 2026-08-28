import os, sys
import numpy as np
import json
import random
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm, trange
import scipy.io
from helpers import *
from MLP import *
#from PIL import Image
import cv2 as cv
import time
import random
import string 
from pyhocon import ConfigFactory
from models.fields import RenderingNetwork, SDFNetwork, SingleVarianceNetwork, NeRF
from models.renderer import NeuSRenderer
import trimesh

import csv
from datetime import datetime, timezone
from scipy.spatial import cKDTree

from itertools import groupby
from operator import itemgetter
from load_data import *
import logging
import argparse 
# import wandb
# from models.testing import RenderNetLamb
import shutil

import NeuS.exp_runner
from device_utils import get_preferred_device

import math

from load_sonarcloud import load_sonarcloud

# set seeds
torch.random.manual_seed(0)
np.random.seed(0)
random.seed(0)

def config_parser():
    import configargparse
    parser = configargparse.ArgumentParser()

# def scatterXYZ(X,Y,Z, name, expID, elev=None, azim=None, lims=None):
#     fig = plt.figure()
#     ax = fig.add_subplot(1,1,1, projection='3d')  
#     #surf = ax.plot_trisurf(X, Y, Z, linewidth=0, antialiased=False)
#     if lims is not None:
#         ax.set_xlim(lims['x_min'], lims['x_max'])
#         ax.set_ylim(lims['y_min'], lims['y_max'])
#         ax.set_zlim(lims['z_min'], lims['z_max'])
#     surf = ax.scatter3D(X.cpu().numpy(), Y.cpu().numpy(), Z.cpu().numpy(), color = "green")
#     if elev is not None and azim is not None:
#         print("Setting elevation and azimuth to {} {}".format(elev, azim))
#         ax.view_init(elev=elev, azim=azim)
#     plt.xlabel('x', fontsize=18)
#     plt.ylabel('y', fontsize=16)
#     plt.savefig("./experiments/{}/figures/scatters/{}.png".format(expID, name))
#     plt.clf()

def make_occ_eval_fn(neusis_runner, render_step_size=0.05):
    def occ_eval_fn(x):
        with torch.no_grad():
            # print(x.shape)
            sdf = neusis_runner.sdf_network.sdf(x)
            inv_s = neusis_runner.deviation_network(torch.zeros([1, 3]))[:, :1].clip(1e-6, 1e6)

            estimated_next_sdf = sdf - render_step_size * 0.5
            estimated_prev_sdf = sdf + render_step_size * 0.5
            prev_cdf = torch.sigmoid(estimated_prev_sdf * inv_s)
            next_cdf = torch.sigmoid(estimated_next_sdf * inv_s)
            p = prev_cdf - next_cdf
            c = prev_cdf
            alpha = ((p + 1e-5) / (c + 1e-5)).view(-1, 1).clip(0.0, 1.0)
            return alpha
    return occ_eval_fn

class Runner:
    def __init__(self, conf, is_continue=False, write_config=True, testing=False, neus_conf=None, use_wandb=True, random_seed=0):
        conf_path = conf
        f = open(conf_path)
        conf_text = f.read()
        conf_name = str(conf_path).split("/")[-1][:-5]
        self.is_continue = is_continue
        self.conf = ConfigFactory.parse_string(conf_text)
        if use_wandb:
            project_name = "testing" if testing else "testing"
            # breakpoint()
            run = wandb.init(
                # Set the project where this run will be logged
                project=project_name,
                # Track hyperparameters and run metadata
                config=self.conf.as_plain_ordered_dict(),
                name=f"{conf_name}-{str(random_seed)}",
                dir="/tmp/"
            )
        self.neus_conf = neus_conf
        self.write_config = write_config
        if random_seed > 0:
            torch.random.manual_seed(random_seed)
            np.random.seed(random_seed)
            random.seed(random_seed)
        self.random_seed = random_seed
        self.use_wandb = use_wandb
        self.conf_path = conf_path

    def set_params(self):
        self.expID = self.conf.get_string('conf.expID') 

        dataset = self.conf.get_string('conf.dataset')
        self.image_setkeyname =  self.conf.get_string('conf.image_setkeyname') 

        self.device = torch.device("cuda") #if torch.cuda.is_available() else torch.device("cpu")
        self.dataset = dataset
        # Training parameters
        self.end_iter = self.conf.get_int('train.end_iter')
        self.N_rand = self.conf.get_int('train.num_select_pixels') #H*W 
        self.arc_n_samples = self.conf.get_int('train.arc_n_samples')
        self.save_freq = self.conf.get_int('train.save_freq')
        self.report_freq = self.conf.get_int('train.report_freq')
        self.val_mesh_freq = self.conf.get_int('train.val_mesh_freq')
        self.learning_rate = self.conf.get_float('train.learning_rate')
        self.learning_rate_alpha = self.conf.get_float('train.learning_rate_alpha')
        self.warm_up_end = self.conf.get_float('train.warm_up_end', default=0.0)
        self.anneal_end = self.conf.get_float('train.anneal_end', default=0.0)
        self.percent_select_true = self.conf.get_float('train.percent_select_true', default=0.5)
        self.r_div = self.conf.get_bool('train.r_div')
        self.train_frac = self.conf.get_float("train.train_frac", default=1.0)
        self.accel = self.conf.get_bool('train.accel', default=False)

         # Metrics
        self.metrics_enabled = self.conf.get_bool(
            "metrics.enabled",
            default=True
        )

        self.metrics_n_samples = self.conf.get_int(
            "metrics.n_samples",
            default=100000
        )

        self.metrics_thresholds = self.conf.get_list(
            "metrics.thresholds",
            default=[0.01, 0.02, 0.05]
        )

        self.metrics_thresholds = [
            float(x) for x in self.metrics_thresholds
        ]

        self.metrics_align = self.conf.get_string(
            "metrics.align",
            default="centroid"
        )

        self.metrics_save_per_mesh = self.conf.get_bool(
            "metrics.save_per_mesh",
            default=True
        )

        self.metrics_save_csv = self.conf.get_bool(
            "metrics.save_csv",
            default=True
        )

        self.gt_geometry_path = self.conf.get_string(
            "metrics.gt_geometry",
            default=""
        )

        # breakpoint()
        self.val_img_freq = self.conf.get_int("train.val_img_freq", default=10000)
        self.lamb_shading = self.conf.get_bool("train.lamb_shading", default=False)
        self.do_weight_norm = self.conf.get_bool("train.do_weight_norm", default=False)
        self.mode_tradeoff_schedule = self.conf.get_string("train.mode_tradeoff_schedule", default="none")
        self.mode_tradeoff_step_iter = self.conf.get_int("train.mode_tradeoff_step_iter", default=-1)
        self.rgb_weight = self.conf.get_float("train.rgb_weight", default=0.0)

        # Weights
        self.igr_weight = self.conf.get_float('train.igr_weight')
        self.variation_reg_weight = self.conf.get_float('train.variation_reg_weight')
        self.px_sample_min_weight = self.conf.get_float('train.px_sample_min_weight')
        # TODO: make below more reasonable? 
        self.weight_sum_factor = self.conf.get_float("train.weight_sum_factor", default=0.1)
        self.dark_weight_sum_factor = self.conf.get_float("train.dark_weight_sum_factor", default=0.0)

        self.ray_n_samples = self.conf['model.neus_renderer']['n_samples']
        # TODO: make below more flexible
        self.base_exp_dir = f"{self.expID}/{self.random_seed}"
        # Diretórios do experimento
        for dirname in [
            "checkpoints",
            "normals",
            "meshes",
            "recordings",
            "validations_fine",
            "metrics",
        ]:
            os.makedirs(
                os.path.join(self.base_exp_dir, dirname),
                exist_ok=True
            )

        self.metrics_dir = os.path.join(
            self.base_exp_dir,
            "metrics"
        )

        os.makedirs(self.metrics_dir, exist_ok=True)
        os.makedirs(self.base_exp_dir, exist_ok=True)
        self.mesh_resolution = self.conf.get_int(
            "mesh.resolution",
            default=64
        )

        shutil.copy(self.conf_path, f"{self.base_exp_dir}/config.conf")
        self.randomize_points = self.conf.get_float('train.randomize_points')
        self.select_px_method = self.conf.get_string('train.select_px_method')
        self.select_valid_px = self.conf.get_bool('train.select_valid_px')        
        self.x_max = self.conf.get_float('mesh.x_max')
        self.x_min = self.conf.get_float('mesh.x_min')
        self.y_max = self.conf.get_float('mesh.y_max')
        self.y_min = self.conf.get_float('mesh.y_min')
        self.z_max = self.conf.get_float('mesh.z_max')
        self.z_min = self.conf.get_float('mesh.z_min')
        self.level_set = self.conf.get_float('mesh.level_set')
       
        if "SonarCloud" in dataset or "sonar_cloud" in dataset.lower():
            from load_sonarcloud import load_sonarcloud
            self.data = load_sonarcloud(dataset)
        else:
            self.data = load_data(dataset)
        self.H, self.W = self.data[self.image_setkeyname][0].shape

        self.r_min = self.data["min_range"]
        self.r_max = self.data["max_range"]
        self.phi_min = -self.data["vfov"]/2
        self.phi_max = self.data["vfov"]/2
        self.vfov = self.data["vfov"]
        self.hfov = self.data["hfov"]


        self.cube_center = torch.tensor([(self.x_max + self.x_min)/2, (self.y_max + self.y_min)/2, (self.z_max + self.z_min)/2], dtype=torch.float32, device=self.device)

        self.timef = self.conf.get_bool('conf.timef')
        self.end_iter = self.conf.get_int('train.end_iter')
        self.start_iter = self.conf.get_int('train.start_iter')
         
        self.object_bbox_min = self.conf.get_list('mesh.object_bbox_min')
        self.object_bbox_max = self.conf.get_list('mesh.object_bbox_max')

        r_increments = []
        self.sonar_resolution = (self.r_max-self.r_min)/self.H
        for i in range(self.H):
            r_increments.append(i*self.sonar_resolution + self.r_min)

        self.r_increments = torch.FloatTensor(r_increments).to(self.device)

        # extrapath = './experiments/{}'.format(self.expID)
        # if not os.path.exists(extrapath):
        #     os.makedirs(extrapath)

        # extrapath = './experiments/{}/checkpoints'.format(self.expID)
        # if not os.path.exists(extrapath):
        #     os.makedirs(extrapath)

        # extrapath = './experiments/{}/model'.format(self.expID)
        # if not os.path.exists(extrapath):
        #     os.makedirs(extrapath)

        # if self.write_config:
        #     with open('./experiments/{}/config.json'.format(self.expID), 'w') as f:
        #         json.dump(self.conf.__dict__, f, indent = 2)

        # Create all image tensors beforehand to speed up process

        self.i_train = np.arange(len(self.data[self.image_setkeyname]))

        self.coords_all = torch.stack(
        torch.meshgrid(
            torch.arange(self.H, device=self.device),
            torch.arange(self.W, device=self.device),
            indexing="ij"
        ), dim=-1 ).reshape(-1, 2)

        self.criterion = torch.nn.L1Loss(reduction='sum')
        
        self.model_list = []
        self.writer = None

        # Networks
        params_to_train = []
        self.sdf_network = SDFNetwork(**self.conf['model.sdf_network']).to(self.device)

        self.deviation_network = SingleVarianceNetwork(**self.conf['model.variance_network']).to(self.device)
        self.color_network = RenderingNetwork(**self.conf['model.rendering_network']).to(self.device)
        params_to_train += list(self.sdf_network.parameters())
        params_to_train += list(self.deviation_network.parameters())
        params_to_train += list(self.color_network.parameters())
        if self.neus_conf is not None:
            neus_runner = NeuS.exp_runner.Runner(self.neus_conf, init_opt=False, sdf_network=self.sdf_network, random_seed=self.random_seed)
            params_to_train += list(neus_runner.nerf_outside.parameters())
            params_to_train += list(neus_runner.deviation_network.parameters()) 
            params_to_train += list(neus_runner.color_network.parameters())  
            self.neus_runner = neus_runner

        self.optimizer = torch.optim.Adam(params_to_train, lr=self.learning_rate)


        self.iter_step = 0
        self.renderer = NeuSRenderer(self.sdf_network,
                                    self.deviation_network,
                                    self.color_network if not self.lamb_shading else RenderNetLamb(),
                                    self.base_exp_dir,
                                    self.expID,
                                    **self.conf['model.neus_renderer'])  

        latest_model_name = None
        if self.is_continue:
            model_list_raw = os.listdir(os.path.join(self.base_exp_dir, 'checkpoints'))
            model_list = []
            for model_name in model_list_raw:
                if model_name[-3:] == 'pth': #and int(model_name[5:-4]) <= self.end_iter:
                    model_list.append(model_name)
            model_list.sort()
            latest_model_name = model_list[-1]

        if latest_model_name is not None:
            logging.info('Find checkpoint: {}'.format(latest_model_name))
            ckpt_dir = os.path.join(self.base_exp_dir, 'checkpoints')
            self.load_checkpoint(f"{ckpt_dir}/{latest_model_name}")

        if self.accel:
            self.occ_eval_fn = make_occ_eval_fn(self) 
            grid_resolution = 128
            grid_nlvl = 1
            device = self.device
            aabb = torch.tensor([-1.0, -1.0, -1.0, 1.0, 1.0, 1.0], device=device)
            self.estimator = OccGridEstimator(
                roi_aabb=aabb, resolution=grid_resolution, levels=grid_nlvl
            ).to(device)
        else:
            self.estimator = None

        self.gt_mesh = self.load_ground_truth_geometry()
        self.gt_points = self.data.get("gt_points", None) if isinstance(self.data, dict) else None
        if self.gt_points is not None:
            logging.info(f"SonarCloud GT point cloud disponível: {self.gt_points.shape[0]} pontos.")
        if self.gt_mesh is not None:
            logging.info(f"GT Mesh disponível: {len(self.gt_mesh.vertices)} vértices.")

    def load_ground_truth_geometry(self):
        """
        Carrega a geometria ground truth uma única vez.

        Retorna um trimesh.Trimesh ou None.
        """

        if not self.gt_geometry_path:
            return None

        if not os.path.exists(self.gt_geometry_path):
            logging.warning(
                "Ground truth não encontrado: %s",
                self.gt_geometry_path
            )
            return None

        try:
            gt = trimesh.load(
                self.gt_geometry_path,
                force="mesh",
                process=False
            )

            logging.info(
                "Ground truth carregado: %s vertices, %s faces",
                len(gt.vertices),
                len(gt.faces)
            )

            return gt

        except Exception as e:
            logging.exception(
                "Erro carregando ground truth: %s",
                e
            )
            return None
        
    def evaluate_mesh_metrics(self, mesh, mesh_path, gt_points=None):
        """
        Calcula métricas da mesh reconstruída contra o Ground Truth (mesh ou nuvem de pontos).

        Sempre calcula estatísticas da mesh.

        Se GT existir (gt_mesh ou gt_points), calcula:
        - Chamfer L1
        - Chamfer L2
        - Accuracy (pred -> GT)
        - Completeness (GT -> pred)
        - Precision, Recall, F1 para cada threshold
        """

        metrics = {
            "iteration": int(self.iter_step),
            "mesh": os.path.relpath(
                mesh_path,
                self.base_exp_dir
            ),
            "timestamp_utc": datetime.now(
                timezone.utc
            ).isoformat(),

            "mesh_stats": {
                "vertices": int(len(mesh.vertices)),
                "faces": int(len(mesh.faces)),
                "is_watertight": bool(mesh.is_watertight),
                "is_winding_consistent": bool(
                    mesh.is_winding_consistent
                ),
            }
        }

        # Bounds
        if len(mesh.vertices) > 0:
            bounds = mesh.bounds
            metrics["mesh_stats"]["bounds_min"] = (
                bounds[0].astype(float).tolist()
            )
            metrics["mesh_stats"]["bounds_max"] = (
                bounds[1].astype(float).tolist()
            )

        # Mesh vazia
        if len(mesh.vertices) == 0:
            metrics["geometry"] = {
                "error": "Predicted mesh has no vertices"
            }
            return metrics

        # Determinar pontos GT para avaliação
        target_gt_pts = gt_points if gt_points is not None else self.gt_points
        n_samples = self.metrics_n_samples

        if self.gt_mesh is not None and len(self.gt_mesh.vertices) > 0:
            gt_eval_points, _ = trimesh.sample.sample_surface(
                self.gt_mesh,
                n_samples
            )
        elif target_gt_pts is not None and len(target_gt_pts) > 0:
            if len(target_gt_pts) > n_samples:
                idx = np.random.choice(len(target_gt_pts), n_samples, replace=False)
                gt_eval_points = target_gt_pts[idx]
            else:
                gt_eval_points = target_gt_pts
        else:
            # Sem GT: salva somente estatísticas da mesh
            return metrics

        if len(gt_eval_points) == 0:
            metrics["geometry"] = {
                "error": "Ground truth has no points"
            }
            return metrics

        # -------------------------------------------------
        # Amostragem de pontos na malha predita
        # -------------------------------------------------
        n_pred_samples = min(n_samples, max(1000, len(mesh.vertices)))
        pred_points, _ = trimesh.sample.sample_surface(
            mesh,
            n_pred_samples
        )

        # -------------------------------------------------
        # Alinhamento de Sistemas de Coordenadas
        # -------------------------------------------------
        align_mode = getattr(self, "metrics_align", "centroid").lower()
        if align_mode in ["centroid", "center"]:
            pred_offset = np.mean(pred_points, axis=0)
            gt_offset = np.mean(gt_eval_points, axis=0)
            pred_eval_pts = pred_points - pred_offset
            gt_eval_pts = gt_eval_points - gt_offset
            metrics["mesh_stats"]["pred_center"] = pred_offset.astype(float).tolist()
            metrics["mesh_stats"]["gt_center"] = gt_offset.astype(float).tolist()
        elif align_mode == "bbox_center":
            pred_offset = (pred_points.max(axis=0) + pred_points.min(axis=0)) / 2.0
            gt_offset = (gt_eval_points.max(axis=0) + gt_eval_points.min(axis=0)) / 2.0
            pred_eval_pts = pred_points - pred_offset
            gt_eval_pts = gt_eval_points - gt_offset
            metrics["mesh_stats"]["pred_center"] = pred_offset.astype(float).tolist()
            metrics["mesh_stats"]["gt_center"] = gt_offset.astype(float).tolist()
        else:
            pred_eval_pts = pred_points
            gt_eval_pts = gt_eval_points

        # -------------------------------------------------
        # KD Trees
        # -------------------------------------------------
        gt_tree = cKDTree(gt_eval_pts)
        pred_tree = cKDTree(pred_eval_pts)

        # Pred -> GT (Accuracy)
        dist_pred_to_gt, _ = gt_tree.query(pred_eval_pts, k=1)
        # GT -> Pred (Completeness)
        dist_gt_to_pred, _ = pred_tree.query(gt_eval_pts, k=1)

        accuracy = float(np.mean(dist_pred_to_gt))
        completeness = float(np.mean(dist_gt_to_pred))

        chamfer_l1 = float(accuracy + completeness)
        chamfer_l2 = float(
            np.mean(dist_pred_to_gt ** 2) +
            np.mean(dist_gt_to_pred ** 2)
        )

        metrics["geometry"] = {
            "alignment": align_mode,
            "accuracy": accuracy,
            "completeness": completeness,
            "chamfer_l1": chamfer_l1,
            "chamfer_l2": chamfer_l2,
            "gt_points_count": int(len(gt_eval_points)),
            "pred_points_count": int(len(pred_points))
        }

        # -------------------------------------------------
        # Precision / Recall / F1
        # -------------------------------------------------
        metrics["thresholds"] = {}
        for threshold in self.metrics_thresholds:
            precision = float(np.mean(dist_pred_to_gt < threshold))
            recall = float(np.mean(dist_gt_to_pred < threshold))

            if precision + recall > 0:
                f1 = float(2.0 * precision * recall / (precision + recall))
            else:
                f1 = 0.0

            threshold_key = f"{threshold:.6f}"
            metrics["thresholds"][threshold_key] = {
                "precision": precision,
                "recall": recall,
                "f1": f1
            }

        return metrics
    def save_mesh_metrics(self, metrics):
        """
        Salva:
        - metrics/XXXXXXXX.json
        - metrics.csv
        - best_metrics.json
        """

        iteration = metrics["iteration"]

        # ---------------------------------------------
        # JSON individual
        # ---------------------------------------------

        json_path = os.path.join(
            self.metrics_dir,
            f"{iteration:08d}.json"
        )

        if self.metrics_save_per_mesh:

            with open(json_path, "w") as f:

                json.dump(
                    metrics,
                    f,
                    indent=2,
                    allow_nan=False
                )

        # ---------------------------------------------
        # CSV consolidado
        # ---------------------------------------------

        if self.metrics_save_csv:

            csv_path = os.path.join(
                self.base_exp_dir,
                "metrics.csv"
            )

            row = {
                "iteration": iteration,
                "mesh": metrics["mesh"],
                "vertices": metrics["mesh_stats"]["vertices"],
                "faces": metrics["mesh_stats"]["faces"],
                "is_watertight": metrics["mesh_stats"]["is_watertight"],
            }

            geometry = metrics.get("geometry", {})

            for key in [
                "accuracy",
                "completeness",
                "chamfer_l1",
                "chamfer_l2"
            ]:
                row[key] = geometry.get(key, "")

            for threshold, values in metrics.get(
                "thresholds",
                {}
            ).items():

                safe_threshold = threshold.replace(".", "_")

                row[
                    f"precision_{safe_threshold}"
                ] = values["precision"]

                row[
                    f"recall_{safe_threshold}"
                ] = values["recall"]

                row[
                    f"f1_{safe_threshold}"
                ] = values["f1"]

            file_exists = os.path.exists(csv_path)

            with open(
                csv_path,
                "a",
                newline=""
            ) as f:

                writer = csv.DictWriter(
                    f,
                    fieldnames=list(row.keys())
                )

                if not file_exists:
                    writer.writeheader()

                writer.writerow(row)

        # ---------------------------------------------
        # Melhor resultado
        # ---------------------------------------------

        self.update_best_metrics(metrics)

        return json_path
    
    def update_best_metrics(self, metrics):
        """
        Atualiza best_metrics.json.

        Critério:
        maior F1 do maior threshold configurado.
        """

        thresholds = metrics.get("thresholds", {})

        if not thresholds:
            return

        # Usa o threshold central, se possível.
        threshold_keys = list(thresholds.keys())

        threshold_keys = sorted(
            threshold_keys,
            key=float
        )

        selected_key = threshold_keys[
            len(threshold_keys) // 2
        ]

        current_f1 = thresholds[
            selected_key
        ]["f1"]

        best_path = os.path.join(
            self.base_exp_dir,
            "best_metrics.json"
        )

        previous_best = None

        if os.path.exists(best_path):

            try:
                with open(best_path, "r") as f:
                    previous_best = json.load(f)

            except Exception:
                previous_best = None

        should_update = (
            previous_best is None or
            current_f1 > previous_best.get("value", -np.inf)
        )

        if should_update:

            best_data = {
                "best_iteration": metrics["iteration"],
                "selection_metric": f"f1_{selected_key}",
                "value": current_f1,
                "mesh": metrics["mesh"],
                "metrics_file": (
                    f"metrics/{metrics['iteration']:08d}.json"
                )
            }

            with open(best_path, "w") as f:
                json.dump(
                    best_data,
                    f,
                    indent=2
                )

    def plotAllArcs(self, use_new=True):
        # This function is used to plot all arc points from all images in the same reference frame
        # this is to verify that we get an approximate shape and that the coordinate transformation 
        # is correct
        i_train = np.arange(len(self.data[self.image_setkeyname]))

        all_points = []
        for j in trange(0, len(i_train)):
            img_i = i_train[j]
            # print(img_i)
            target = self.data[self.image_setkeyname][img_i]
            target = torch.Tensor(target).to(self.device)
            pose = self.data["sensor_poses"][img_i]
            c2w = torch.tensor(pose, dtype=torch.float32, device=self.device)
            coords = torch.nonzero(target)
            n_pixels = len(coords)
            if n_pixels == 0: continue
            # TODO: do something faster than below 
            # essentially, just get dirs before normalization from the get_arcs function 

            # old
            if not use_new:
                _, _, _, _, pts, _ = get_arcs(self.H, self.W, self.phi_min, self.phi_max, self.r_min, self.r_max,  torch.tensor(pose, dtype=torch.float32, device=self.device), n_pixels,
                                                        self.arc_n_samples, self.ray_n_samples, self.hfov, coords,
                                                        self.r_increments, self.randomize_points, 
                                                        self.device, self.cube_center)
                pts = pts.reshape(n_pixels, self.arc_n_samples, self.ray_n_samples, 3)
                pts = pts[:, :, -1, :]
                # print(pts.shape)
                all_points.append(pts)
            else:
                img_y = coords[:, 0] # img y coords
                img_x = coords[:, 1] # img x coords 
                phi = (
                    torch.linspace(self.phi_min, self.phi_max, self.arc_n_samples)
                    .float()
                    .repeat(n_pixels)
                    .reshape(n_pixels, -1)
                )
                sonar_resolution = (self.r_max - self.r_min) / self.H
                # compute radius at each pixel
                r = img_y * sonar_resolution + self.r_min
                # compute bearing angle at each pixel (azimuth)
                theta = -self.hfov / 2 + img_x * self.hfov / self.W
                coords = torch.stack(
                    (
                        r.repeat_interleave(self.arc_n_samples).reshape(n_pixels, -1),
                        theta.repeat_interleave(self.arc_n_samples).reshape(n_pixels, -1),
                        phi,
                    ),
                    dim=-1,
                )
                coords = coords.reshape(-1, 3)
                X = coords[:, 0] * torch.cos(coords[:, 1]) * torch.cos(coords[:, 2])
                Y = coords[:, 0] * torch.sin(coords[:, 1]) * torch.cos(coords[:, 2])
                Z = coords[:, 0] * torch.sin(coords[:, 2])
                pts = c2w @ torch.stack((X, Y, Z, torch.ones_like(X)))
                pts = pts[:3, ...].T

                all_points.append(pts)
        
        all_points = torch.cat(all_points, dim=0)
        return all_points, target
    
    def getRandomImgCoordsByPercentage(self, target):

        target = torch.as_tensor(
            target,
            dtype=torch.float32,
            device=self.device
        )

        # --------------------------------------------------
        # Pixels com intensidade > 0
        # --------------------------------------------------
        true_coords = torch.nonzero(target > 0)

        # --------------------------------------------------
        # Random pixels
        # --------------------------------------------------
        N_rand = self.N_rand

        rand_y = torch.randint(
            0,
            self.H,
            (N_rand,),
            device=self.device
        )

        rand_x = torch.randint(
            0,
            self.W,
            (N_rand,),
            device=self.device
        )

        random_coords = torch.stack(
            [rand_y, rand_x],
            dim=1
        )

        # --------------------------------------------------
        # Seleciona pixels positivos
        # --------------------------------------------------
        if len(true_coords) > 0:

            n_true = int(
                self.percent_select_true * len(true_coords)
            )

            n_true = min(
                n_true,
                len(true_coords)
            )

            if n_true > 0:
                perm = torch.randperm(
                    len(true_coords),
                    device=self.device
                )[:n_true]

                true_coords = true_coords[perm]

        # --------------------------------------------------
        # Junta
        # --------------------------------------------------
        coords = torch.cat(
            [
                random_coords,
                true_coords
            ],
            dim=0
        )

        # --------------------------------------------------
        # LIMITA O TOTAL
        # --------------------------------------------------
        max_pixels = 1000

        if len(coords) > max_pixels:

            perm = torch.randperm(
                len(coords),
                device=self.device
            )[:max_pixels]

            coords = coords[perm]

        return coords, target

    def train(self):
        loss_arr = []
        self.validate_mesh(threshold = self.level_set)

        # make train/validation sets
        # fix validation set for fair comparisons 
        # i_all = np.arange(len(self.data[self.image_setkeyname]))
        i_train = [] 
        i_val = []
        if self.train_frac < 1.0: 
            val_skip = int(1 / (1 - self.train_frac))
            for i in range(len(self.data[self.image_setkeyname])):
                if i % val_skip == 0:
                    i_val.append(i)
                else:
                    i_train.append(i)
        else:
            i_train = np.arange(len(self.data[self.image_setkeyname]))
            i_val = [] 
        # np.random.shuffle(i_all)
        # split_ind = int(self.train_frac * len(i_all))
        # i_train = i_all[:split_ind]
        # i_val = i_all[split_ind:]

        for i in trange(self.start_iter, self.end_iter):
            # i_train = np.arange(len(self.data[self.image_setkeyname]))
            np.random.shuffle(i_train)
            loss_total = 0
            sum_intensity_loss = 0
            sum_eikonal_loss = 0
            sum_total_variational = 0
            sum_neus_loss = 0 
            
            for j in trange(0, len(i_train)):
                if self.accel:
                    self.estimator.update_every_n_steps(step=self.iter_step, occ_eval_fn=self.occ_eval_fn)
                log_dict = {}
                img_i = i_train[j]
                target = self.data[self.image_setkeyname][img_i]

                
                pose = self.data["sensor_poses"][img_i]  
                
                if self.select_px_method == "byprob":
                    coords, target = self.getRandomImgCoordsByProbability(target)
                else:
                    coords, target = self.getRandomImgCoordsByPercentage(target)

                n_pixels = len(coords)
                print(
                    f"n_pixels={n_pixels}, "
                    f"arc_n_samples={self.arc_n_samples}, "
                    f"ray_n_samples={self.ray_n_samples}"
                )
                # r holds radius per sample if estimator is none, otherwise it is  nONe
                rays_d, dphi, r, rs, pts, dists = get_arcs(self.H, self.W, self.phi_min, self.phi_max, self.r_min, self.r_max,  torch.tensor(pose, dtype=torch.float32, device=self.device), n_pixels,
                                                        self.arc_n_samples, self.ray_n_samples, self.hfov, coords, self.r_increments, 
                                                        self.randomize_points, self.device, self.cube_center, self.estimator)

                
                target_s = target[coords[:, 0], coords[:, 1]]

                render_out = self.renderer.render_sonar(rays_d, pts, dists, n_pixels, 
                                                        self.arc_n_samples, self.ray_n_samples, r,
                                                        cos_anneal_ratio=self.get_cos_anneal_ratio())


                gradient_error = render_out['gradient_error'].reshape(-1, 1) #.reshape(n_pixels, self.arc_n_samples, -1)
                # gradient_error = torch.tensor(0)
                eikonal_loss = gradient_error.sum()*(1/gradient_error.shape[0])
                variation_regularization = render_out['variation_error']*(1/(self.arc_n_samples*self.ray_n_samples*n_pixels))
                # variation_regularization = torch.tensor(0)

                # try bright weight sum regularization 
                if self.weight_sum_factor > 0.0:
                    bright_weight_sums = render_out["weight_sum"][target_s > 0.0]
                    ones_target = torch.ones_like(bright_weight_sums) 
                    # modified with max
                    # weight_norm_loss = self.weight_sum_factor * torch.mean((torch.maximum(ones_target-bright_weight_sums, torch.zeros_like(ones_target)))**2)
                    weight_norm_loss = self.weight_sum_factor * torch.mean((ones_target-bright_weight_sums)**2)
                else:
                    weight_norm_loss = torch.tensor(0.0)

                # weight sparsity regularization 
                # bright_weights = render_out["weights"][target_s > 0.0]
                # weight_sparse_loss = 0.1 * torch.nn.functional.l1_loss(bright_weights, torch.zeros_like(bright_weights))
                weight_sparse_loss = 0.0

                # dark weight sum regularization
                # breakpoint()
                if self.dark_weight_sum_factor > 0.0:
                    dark_weights = render_out["weight_sum"][target_s == 0.0]
                    zeros_target = torch.zeros_like(dark_weights)
                    dark_weight_norm_loss = self.dark_weight_sum_factor * torch.mean((dark_weights - zeros_target)**2)
                else:
                    dark_weight_norm_loss = torch.tensor(0.0)

                if self.r_div:
                    intensityPointsOnArc = render_out["intensityPointsOnArc"]
                    intensity_fine = (torch.divide(intensityPointsOnArc, rs)*render_out["weights"]).sum(dim=1) 
                else:
                    intensity_fine = render_out['color_fine']

                if self.do_weight_norm:
                    if len(intensity_fine.shape) == 1:
                        intensity_fine = intensity_fine[:, None]
                    intensity_fine[target_s > 0.0] = intensity_fine[target_s > 0.0] / render_out["weight_sum"][target_s > 0.0]

                intensity_error = self.criterion(intensity_fine.squeeze(), target_s.squeeze())*(1/n_pixels)
                
                loss = intensity_error + eikonal_loss * self.igr_weight  + variation_regularization*self.variation_reg_weight
                loss += weight_norm_loss 
                loss += weight_sparse_loss
                loss += dark_weight_norm_loss
                if self.neus_conf is not None: 
                    if self.mode_tradeoff_schedule == "step":
                        if self.iter_step < self.mode_tradeoff_step_iter:
                            neus_loss = torch.tensor([0.]) 
                        else:
                            neus_loss = self.neus_runner.do_one_iter(img_i % self.neus_runner.dataset.n_images) 
                            loss = (1 - self.rgb_weight) * loss + self.rgb_weight * neus_loss
                    else:
                        neus_loss = self.neus_runner.do_one_iter(img_i % self.neus_runner.dataset.n_images) 
                        loss += neus_loss * 2 # TODO: fix this (add config?)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                with torch.no_grad():
                    lossNG = intensity_error + eikonal_loss * self.igr_weight 
                    loss_total += lossNG.cpu().numpy().item()
                    sum_intensity_loss += intensity_error.cpu().numpy().item()
                    sum_eikonal_loss += eikonal_loss.cpu().numpy().item()
                    sum_total_variational +=  variation_regularization.cpu().numpy().item()
                    if self.neus_conf is not None:
                        sum_neus_loss += neus_loss.cpu().numpy().item()
                
                
                self.iter_step += 1
                self.update_learning_rate()

                del(target)
                del(target_s)
                del(rays_d)
                del(pts)
                del(dists)
                del(render_out)
                del(coords)
                # break
                log_dict["sonar_intensity_loss"] = intensity_error.item()

                # end of epoch
                if j == len(i_train) - 1:
                    epoch_num = i // len(self.data[self.image_setkeyname]) # duplicated with below
                    log_dict["epoch_sonar_intensity_loss"] = sum_intensity_loss/len(i_train)
                    log_dict["epoch_num"] = epoch_num
                    if (epoch_num+1) % self.val_img_freq == 0:
                        tqdm.write("validation\n")
                        val_metric = 0
                        for i in trange(len(i_val)):
                            val_ind = i_val[i]
                            curr_img_val = render_image(self, val_ind, self.estimator)
                            curr_gt_val = self.data[self.image_setkeyname][val_ind]
                            val_metric += np.mean((curr_img_val - curr_gt_val) ** 2)
                        val_metric = val_metric / len(i_val) 
                        log_dict["mean_val_mse"] = val_metric

                        img = render_image(self, i_val[len(i_val)//2], self.estimator)
                        if self.use_wandb:
                            log_dict["val_vis"] = wandb.Image((np.clip(img, 0, 1)*255).astype(np.uint8))
                        img_train = render_image(self, i_train[len(i_train)//2], self.estimator)
                        if self.use_wandb:
                            log_dict["train_vis"] = wandb.Image((np.clip(img_train, 0, 1)*255).astype(np.uint8))
                        train_gt_img = self.data[self.image_setkeyname][i_train[len(i_train)//2]]
                        if self.use_wandb:
                            log_dict["train_gt_vis"] = wandb.Image((np.clip(train_gt_img, 0, 1)*255).astype(np.uint8))
                        gt_img = self.data[self.image_setkeyname][i_val[len(i_val)//2]]
                        if self.use_wandb:
                            log_dict["val_gt_vis"] = wandb.Image((np.clip(gt_img, 0, 1)*255).astype(np.uint8))
                        log_dict["epoch_num_val"] = epoch_num

                    # saving mesh + novel view synthesis for neus
                    if epoch_num == 0 or epoch_num % self.val_mesh_freq == 0:
                        mesh_path = self.validate_mesh(threshold = self.level_set)
                        if self.neus_conf is not None: 
                            self.neus_runner.validate_mesh() 
                            self.neus_runner.validate_image()
                        if self.use_wandb:
                            log_dict["mesh_recon"] = wandb.Object3D(open(mesh_path))
                if self.use_wandb:
                    wandb.log(log_dict)

            with torch.no_grad():
                l = loss_total/len(i_train)
                iL =  sum_intensity_loss/len(i_train)
                eikL =  sum_eikonal_loss/len(i_train)
                varL =  sum_total_variational/len(i_train)
                if self.neus_conf is not None:
                    nl = sum_neus_loss / len(i_train)
                loss_arr.append(l)
            # breakpoint()
            epoch_num = i // len(self.data[self.image_setkeyname])

            # saving checkpoint
            if epoch_num == 0 or epoch_num % self.save_freq == 0:
                logging.info('iter:{} ********************* SAVING CHECKPOINT ****************'.format(self.optimizer.param_groups[0]['lr']))
                self.save_checkpoint()
                if self.neus_conf is not None: 
                    self.neus_runner.save_checkpoint()
            
            # write to terminal
            if epoch_num % self.report_freq == 0:
                report_str = f"iter:{self.iter_step:8>d} Loss={l} | intensity Loss={iL}  | eikonal loss={eikL} | total variation loss = {varL} | lr = {self.optimizer.param_groups[0]['lr']}"
                if self.neus_conf is not None: 
                    report_str = f"{report_str} | neus loss = {nl}"
                report_str = f"{report_str} | weight_norm_loss = {weight_norm_loss.item()}"
                report_str = f"{report_str} | dark_weight_norm_loss = {dark_weight_norm_loss.item()}"
                # report_str = f"{report_str} | weight_sparse_loss = {weight_sparse_loss.item()}"
                # print(report_str)
                tqdm.write(report_str)
        
        self.save_checkpoint()
        self.validate_mesh(threshold = self.level_set)


    def save_checkpoint(self):
        checkpoint = {
            'sdf_network_fine': self.sdf_network.state_dict(),
            'variance_network_fine': self.deviation_network.state_dict(),
            'color_network_fine': self.color_network.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'iter_step': self.iter_step,
        }

        os.makedirs(os.path.join(self.base_exp_dir, 'checkpoints'), exist_ok=True)
        torch.save(checkpoint, os.path.join(self.base_exp_dir, 'checkpoints', 'ckpt_{:0>6d}.pth'.format(self.iter_step)))

    def load_checkpoint(self, checkpoint_name):
        # checkpoint = torch.load(os.path.join(self.base_exp_dir, 'checkpoints', checkpoint_name), map_location=self.device)
        checkpoint = torch.load(checkpoint_name, map_location=self.device, weights_only=False)

        # Depois de carregar, force movimento
        self.sdf_network.load_state_dict(checkpoint['sdf_network_fine'])
        self.sdf_network = self.sdf_network.to(self.device)

        self.sdf_network.load_state_dict(checkpoint['sdf_network_fine'])
        self.deviation_network.load_state_dict(checkpoint['variance_network_fine'])
        self.color_network.load_state_dict(checkpoint['color_network_fine'])

        # Tenta carregar optimizer state; se falhar (parameter groups mudaram),
        # reinicia o optimizer mantendo os pesos dos modelos.
        try:
            self.optimizer.load_state_dict(checkpoint['optimizer'])
        except ValueError as e:
            print(f'[load_checkpoint] Falha ao carregar optimizer state ({e}); reiniciando optimizer.')
            self.iter_step = 0
            return

        self.iter_step = checkpoint['iter_step']

    def update_learning_rate(self):
        if self.iter_step <= self.warm_up_end: # do i really need <=?
            learning_factor = self.iter_step / self.warm_up_end
        else:
            alpha = self.learning_rate_alpha
            progress = (self.iter_step - self.warm_up_end) / (self.end_iter - self.warm_up_end)
            learning_factor = (np.cos(np.pi * progress) + 1.0) * 0.5 * (1 - alpha) + alpha

        for g in self.optimizer.param_groups:
            g['lr'] = self.learning_rate * learning_factor

    def get_cos_anneal_ratio(self):
        if self.anneal_end == 0.0:
            return 1.0
        else:
            return np.min([1.0, self.iter_step / self.anneal_end])
    
    def validate_mesh(
        self,
        world_space=False,
        resolution=None,
        threshold=0.0
    ):
        if resolution is None:
            resolution = self.mesh_resolution

        bound_min = torch.tensor(
            self.object_bbox_min,
            dtype=torch.float32
        )

        bound_max = torch.tensor(
            self.object_bbox_max,
            dtype=torch.float32
        )

        self.sdf_network = self.sdf_network.to(
            self.device
        )

        self.deviation_network = (
            self.deviation_network.to(self.device)
        )

        self.color_network = self.color_network.to(
            self.device
        )

        self.renderer.sdf_network = self.sdf_network
        self.renderer.deviation_network = (
            self.deviation_network
        )
        self.renderer.color_network = (
            self.color_network
        )

        vertices, triangles = (
            self.renderer.extract_geometry(
                bound_min,
                bound_max,
                resolution=resolution,
                threshold=threshold
            )
        )

        mesh_dir = os.path.join(
            self.base_exp_dir,
            "meshes"
        )

        os.makedirs(
            mesh_dir,
            exist_ok=True
        )

        if world_space:
            vertices = (
                vertices *
                self.dataset.scale_mats_np[0][0, 0]
                +
                self.dataset.scale_mats_np[0][:3, 3][None]
            )

        mesh = trimesh.Trimesh(
            vertices=vertices,
            faces=triangles,
            process=False
        )

        mesh_path = os.path.join(
            mesh_dir,
            "{:08d}.obj".format(self.iter_step)
        )

        mesh.export(mesh_path)

        # -----------------------------------------
        # Avaliar e salvar métricas de reconstrução
        # -----------------------------------------
        metrics = self.evaluate_mesh_metrics(
            mesh,
            mesh_path,
            gt_points=self.gt_points
        )

        self.save_mesh_metrics(metrics)

        if "geometry" in metrics and "chamfer_l1" in metrics["geometry"]:
            geom = metrics["geometry"]
            tqdm.write(
                f"\n[VALIDAÇÃO iter {self.iter_step}] Chamfer L1: {geom['chamfer_l1']:.4f} | "
                f"Accuracy: {geom['accuracy']:.4f} | Completeness: {geom['completeness']:.4f}"
            )
            if self.use_wandb:
                wandb_metrics = {
                    "val_mesh/chamfer_l1": geom["chamfer_l1"],
                    "val_mesh/chamfer_l2": geom["chamfer_l2"],
                    "val_mesh/accuracy": geom["accuracy"],
                    "val_mesh/completeness": geom["completeness"],
                }
                for thresh_k, vals in metrics.get("thresholds", {}).items():
                    wandb_metrics[f"val_mesh/f1_{thresh_k}"] = vals["f1"]
                wandb.log(wandb_metrics)

        return mesh_path


if __name__=='__main__':
    # Keep tensor defaults on the standard CPU dtype; device placement is explicit.
    FORMAT = "[%(filename)s:%(lineno)s - %(funcName)20s() ] %(message)s"
    logging.getLogger('matplotlib.font_manager').disabled = True
    logging.basicConfig(level=logging.DEBUG, format=FORMAT)
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--conf', type=str, default="./confs/conf.conf")
    parser.add_argument('--neus_conf', type=str)
    parser.add_argument('--is_continue', default=False, action="store_true")
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument("--testing", action="store_true")
    parser.add_argument("--random_seed", type=int, default=0)
    parser.add_argument("--disable_wandb", action="store_true")

    args = parser.parse_args()

    torch.cuda.set_device(args.gpu)
    runner = Runner(args.conf, args.is_continue, testing=args.testing, neus_conf=args.neus_conf, random_seed=args.random_seed, use_wandb=not args.disable_wandb)
    runner.set_params()
    runner.train()
