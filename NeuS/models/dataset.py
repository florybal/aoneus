import torch
import torch.nn.functional as F
import cv2 as cv
import numpy as np
import os
from glob import glob
from icecream import ic
from scipy.spatial.transform import Rotation as Rot
from scipy.spatial.transform import Slerp
from device_utils import get_preferred_device


# This function is borrowed from IDR: https://github.com/lioryariv/idr
def load_K_Rt_from_P(filename, P=None):
    if P is None:
        lines = open(filename).read().splitlines()
        if len(lines) == 4:
            lines = lines[1:]
        lines = [[x[0], x[1], x[2], x[3]] for x in (x.split(" ") for x in lines)]
        P = np.asarray(lines).astype(np.float32).squeeze()

    out = cv.decomposeProjectionMatrix(P)
    K = out[0]
    R = out[1]
    t = out[2]

    K = K / K[2, 2]
    intrinsics = np.eye(4)
    intrinsics[:3, :3] = K

    pose = np.eye(4, dtype=np.float32)
    pose[:3, :3] = R.transpose()
    pose[:3, 3] = (t[:3] / t[3])[:, 0]

    return intrinsics, pose

def cvt_neus_coords_to_ho_coords(pts):
    """
    args:
        pts: (..., 3)
    """
    original_shape = pts.shape 
    pts = pts.reshape(-1, 3) 
    cmat = torch.tensor(
        [[0.0, 0.0, 1.0], 
        [-1.0, 0.0, 0.0], 
        [0.0, -1.0, 0.0]],
        device=pts.device,
        dtype=pts.dtype,
    )
    out = pts @ cmat.T
    return out.reshape(original_shape)


def _find_image_dir(data_dir):
    for candidate in ("images_wb", "Images_wb", "image", "images"):
        candidate_path = os.path.join(data_dir, candidate)
        if os.path.isdir(candidate_path):
            return candidate_path
    return None


def _load_image_files(image_dir):
    image_paths = []
    for pattern in ("*.png", "*.jpg", "*.jpeg", "*.JPG", "*.JPEG"):
        image_paths.extend(glob(os.path.join(image_dir, pattern)))
    return sorted(image_paths)

class Dataset:
    def __init__(self, conf):
        super(Dataset, self).__init__()
        print('Load data: Begin')
        self.device = get_preferred_device()
        self.conf = conf

        self.data_dir = conf.get_string('data_dir')
        self.camera_outside_sphere = conf.get_bool('camera_outside_sphere', default=True)
        self.scale_mat_scale = conf.get_float('scale_mat_scale', default=1.1)

        self.is_seathru = os.path.exists(os.path.join(self.data_dir, 'poses_bounds.npy')) and _find_image_dir(self.data_dir) is not None

        if self.is_seathru:
            self._load_seathru_dataset()
            print('Load data: End')
            return

        self.render_cameras_name = conf.get_string('render_cameras_name', default='cameras_sphere.npz')
        self.object_cameras_name = conf.get_string('object_cameras_name', default='cameras_sphere.npz')

        camera_dict = np.load(os.path.join(self.data_dir, self.render_cameras_name))
        self.camera_dict = camera_dict
        self.images_lis = sorted(glob(os.path.join(self.data_dir, 'image/*.png')))
        self.n_images = len(self.images_lis)
        self.images_np = np.stack([cv.imread(im_name) for im_name in self.images_lis]) / 256.0
        self.masks_lis = sorted(glob(os.path.join(self.data_dir, 'mask/*.png')))
        self.masks_np = np.stack([cv.imread(im_name) for im_name in self.masks_lis]) / 256.0

        # world_mat is a projection matrix from world to image
        self.world_mats_np = [camera_dict['world_mat_%d' % idx].astype(np.float32) for idx in range(self.n_images)]

        self.scale_mats_np = []

        # scale_mat: used for coordinate normalization, we assume the scene to render is inside a unit sphere at origin.
        self.scale_mats_np = [camera_dict['scale_mat_%d' % idx].astype(np.float32) for idx in range(self.n_images)]

        self.intrinsics_all = []
        self.pose_all = []

        for scale_mat, world_mat in zip(self.scale_mats_np, self.world_mats_np):
            P = world_mat @ scale_mat
            P = P[:3, :4]
            intrinsics, pose = load_K_Rt_from_P(None, P)
            self.intrinsics_all.append(torch.from_numpy(intrinsics).float())
            self.pose_all.append(torch.from_numpy(pose).float())

        self.images = torch.from_numpy(self.images_np.astype(np.float32)).cpu()  # [n_images, H, W, 3]
        self.masks  = torch.from_numpy(self.masks_np.astype(np.float32)).cpu()   # [n_images, H, W, 3]
        self.intrinsics_all = torch.stack(self.intrinsics_all).to(self.device)   # [n_images, 4, 4]
        self.intrinsics_all_inv = torch.inverse(self.intrinsics_all)  # [n_images, 4, 4]
        self.focal = self.intrinsics_all[0][0, 0]
        self.pose_all = torch.stack(self.pose_all).to(self.device)  # [n_images, 4, 4]
        self.H, self.W = self.images.shape[1], self.images.shape[2]
        self.image_pixels = self.H * self.W

        object_bbox_min = np.array([-1.01, -1.01, -1.01, 1.0])
        object_bbox_max = np.array([ 1.01,  1.01,  1.01, 1.0])
        # Object scale mat: region of interest to **extract mesh**
        object_scale_mat = np.load(os.path.join(self.data_dir, self.object_cameras_name))['scale_mat_0']
        object_bbox_min = np.linalg.inv(self.scale_mats_np[0]) @ object_scale_mat @ object_bbox_min[:, None]
        object_bbox_max = np.linalg.inv(self.scale_mats_np[0]) @ object_scale_mat @ object_bbox_max[:, None]
        self.object_bbox_min = object_bbox_min[:3, 0]
        self.object_bbox_max = object_bbox_max[:3, 0]

        print('Load data: End')

    def _load_seathru_dataset(self):
        poses_bounds = np.load(os.path.join(self.data_dir, 'poses_bounds.npy')).astype(np.float32)
        poses = poses_bounds[:, :-2].reshape([-1, 3, 5])
        bds = poses_bounds[:, -2:]

        image_dir = _find_image_dir(self.data_dir)
        self.images_lis = _load_image_files(image_dir)
        if not self.images_lis:
            raise FileNotFoundError(f'No images found in {image_dir}')

        self.n_images = len(self.images_lis)
        self.images_np = np.stack([cv.imread(im_name) for im_name in self.images_lis]) / 256.0
        self.masks_np = np.ones_like(self.images_np)

        H, W, focal = poses[0, :, 4]
        self.H = int(round(float(H)))
        self.W = int(round(float(W)))

        scale = 1.0 / max(1.0, float(np.min(bds) * 0.75))

        self.intrinsics_all = []
        self.pose_all = []
        for pose in poses:
            intrinsics = np.eye(4, dtype=np.float32)
            intrinsics[0, 0] = float(focal)
            intrinsics[1, 1] = float(focal)
            intrinsics[0, 2] = float(W) * 0.5
            intrinsics[1, 2] = float(H) * 0.5

            c2w = np.eye(4, dtype=np.float32)
            c2w[:3, :4] = pose[:3, :4]
            c2w[:3, 3] *= scale

            self.intrinsics_all.append(torch.from_numpy(intrinsics).float())
            self.pose_all.append(torch.from_numpy(c2w).float())

        self.images = torch.from_numpy(self.images_np.astype(np.float32)).cpu()
        self.masks = torch.from_numpy(self.masks_np.astype(np.float32)).cpu()
        self.intrinsics_all = torch.stack(self.intrinsics_all).to(self.device)
        self.intrinsics_all_inv = torch.inverse(self.intrinsics_all)
        self.focal = self.intrinsics_all[0][0, 0]
        self.pose_all = torch.stack(self.pose_all).to(self.device)
        self.image_pixels = self.H * self.W

        camera_centers = self.pose_all[:, :3, 3].detach().cpu().numpy()
        scene_radius = float(np.linalg.norm(camera_centers, axis=1).max())
        bbox_radius = max(1.01, scene_radius * 2.0)
        self.object_bbox_min = np.array([-bbox_radius, -bbox_radius, -bbox_radius], dtype=np.float32)
        self.object_bbox_max = np.array([bbox_radius, bbox_radius, bbox_radius], dtype=np.float32)

        self.camera_dict = None
        self.world_mats_np = []
        self.scale_mats_np = []

    def gen_rays_at(self, img_idx, resolution_level=1):
        """
        Generate rays at world space from one camera.
        """
        l = resolution_level
        tx = torch.linspace(0, self.W - 1, self.W // l)
        ty = torch.linspace(0, self.H - 1, self.H // l)
        pixels_x, pixels_y = torch.meshgrid(tx, ty)
        p = torch.stack([pixels_x, pixels_y, torch.ones_like(pixels_y)], dim=-1) # W, H, 3
        p = torch.matmul(self.intrinsics_all_inv[img_idx, None, None, :3, :3], p[:, :, :, None]).squeeze()  # W, H, 3
        rays_v = p / torch.linalg.norm(p, ord=2, dim=-1, keepdim=True)  # W, H, 3
        rays_v = torch.matmul(self.pose_all[img_idx, None, None, :3, :3], rays_v[:, :, :, None]).squeeze()  # W, H, 3
        rays_o = self.pose_all[img_idx, None, None, :3, 3].expand(rays_v.shape)  # W, H, 3
        rays_o = cvt_neus_coords_to_ho_coords(rays_o) 
        rays_v = cvt_neus_coords_to_ho_coords(rays_v)
        return rays_o.transpose(0, 1), rays_v.transpose(0, 1)

    def gen_random_rays_at(self, img_idx, batch_size):
        """
        Generate random rays at world space from one camera.
        """
        pixels_x = torch.randint(low=0, high=self.W, size=[batch_size])
        pixels_y = torch.randint(low=0, high=self.H, size=[batch_size])
        color = self.images[img_idx].to(self.device)[(pixels_y, pixels_x)]    # batch_size, 3
        mask = self.masks[img_idx].to(self.device)[(pixels_y, pixels_x)]      # batch_size, 3
        p = torch.stack([pixels_x, pixels_y, torch.ones_like(pixels_y)], dim=-1).float()  # batch_size, 3
        p = torch.matmul(self.intrinsics_all_inv[img_idx, None, :3, :3], p[:, :, None]).squeeze() # batch_size, 3
        rays_v = p / torch.linalg.norm(p, ord=2, dim=-1, keepdim=True)    # batch_size, 3
        rays_v = torch.matmul(self.pose_all[img_idx, None, :3, :3], rays_v[:, :, None]).squeeze()  # batch_size, 3
        rays_o = self.pose_all[img_idx, None, :3, 3].expand(rays_v.shape) # batch_size, 3
            # switch to holoocean coordinates 
        rays_v = cvt_neus_coords_to_ho_coords(rays_v)
        rays_o = cvt_neus_coords_to_ho_coords(rays_o)
        return torch.cat([rays_o, rays_v, color, mask[:, :1]], dim=-1).to(self.device)    # batch_size, 10

    def gen_rays_between(self, idx_0, idx_1, ratio, resolution_level=1):
        """
        Interpolate pose between two cameras.
        """
        l = resolution_level
        tx = torch.linspace(0, self.W - 1, self.W // l)
        ty = torch.linspace(0, self.H - 1, self.H // l)
        pixels_x, pixels_y = torch.meshgrid(tx, ty)
        p = torch.stack([pixels_x, pixels_y, torch.ones_like(pixels_y)], dim=-1)  # W, H, 3
        p = torch.matmul(self.intrinsics_all_inv[0, None, None, :3, :3], p[:, :, :, None]).squeeze()  # W, H, 3
        rays_v = p / torch.linalg.norm(p, ord=2, dim=-1, keepdim=True)  # W, H, 3
        trans = self.pose_all[idx_0, :3, 3] * (1.0 - ratio) + self.pose_all[idx_1, :3, 3] * ratio
        pose_0 = self.pose_all[idx_0].detach().cpu().numpy()
        pose_1 = self.pose_all[idx_1].detach().cpu().numpy()
        pose_0 = np.linalg.inv(pose_0)
        pose_1 = np.linalg.inv(pose_1)
        rot_0 = pose_0[:3, :3]
        rot_1 = pose_1[:3, :3]
        rots = Rot.from_matrix(np.stack([rot_0, rot_1]))
        key_times = [0, 1]
        slerp = Slerp(key_times, rots)
        rot = slerp(ratio)
        pose = np.diag([1.0, 1.0, 1.0, 1.0])
        pose = pose.astype(np.float32)
        pose[:3, :3] = rot.as_matrix()
        pose[:3, 3] = ((1.0 - ratio) * pose_0 + ratio * pose_1)[:3, 3]
        pose = np.linalg.inv(pose)
        rot = torch.from_numpy(pose[:3, :3]).to(self.device)
        trans = torch.from_numpy(pose[:3, 3]).to(self.device)
        rays_v = torch.matmul(rot[None, None, :3, :3], rays_v[:, :, :, None]).squeeze()  # W, H, 3
        rays_o = trans[None, None, :3].expand(rays_v.shape)  # W, H, 3
        rays_o = cvt_neus_coords_to_ho_coords(rays_o) 
        rays_v = cvt_neus_coords_to_ho_coords(rays_v)
        return rays_o.transpose(0, 1), rays_v.transpose(0, 1)

    def near_far_from_sphere(self, rays_o, rays_d):
        a = torch.sum(rays_d**2, dim=-1, keepdim=True)
        b = 2.0 * torch.sum(rays_o * rays_d, dim=-1, keepdim=True)
        mid = 0.5 * (-b) / a
        near = mid - 1.0
        far = mid + 1.0
        return near, far

    def image_at(self, idx, resolution_level):
        img = cv.imread(self.images_lis[idx])
        return (cv.resize(img, (self.W // resolution_level, self.H // resolution_level))).clip(0, 255)

