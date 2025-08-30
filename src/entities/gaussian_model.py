#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#
from pathlib import Path

import numpy as np
import open3d as o3d
import torch
from plyfile import PlyData, PlyElement
from simple_knn._C import distCUDA2
from torch import nn

from src.utils.gaussian_model_utils import (RGB2SH, build_scaling_rotation,
                                            get_expon_lr_func, inverse_sigmoid,
                                            strip_symmetric, BasicPointCloud)

class GaussianModel:
    def __init__(self, sh_degree: int = 3, isotropic=False, device:int=0, verbose:bool=False):
        self.gaussian_param_names = [
            "active_sh_degree",
            "xyz",
            "features_dc",
            "features_rest",
            "scaling",
            "rotation",
            "opacity",
            "max_radii2D",
            "xyz_gradient_accum",
            "denom",
            "spatial_lr_scale",
            "optimizer",
        ]
        self.device = device
        self.VERBOSE:bool = verbose
        self.max_sh_degree = sh_degree
        self.active_sh_degree = sh_degree  # temp
        self._xyz = torch.empty(0, device=device)
        self._features_dc = torch.empty(0, device=device)
        self._features_rest = torch.empty(0, device=device)
        self._scaling = torch.empty(0, device=device)
        self._rotation = torch.empty((0, 4), device=device)
        self._opacity = torch.empty(0, device=device)
        self.max_radii2D = torch.empty(0)
        self.xyz_gradient_accum = torch.empty(0)
        self.denom = torch.empty(0)
        self.optimizer = None
        self.scheduler = None
        self.percent_dense = 0
        self.spatial_lr_scale = 1
        self.setup_functions()
        self.isotropic = isotropic

    def restore_from_params(self, params_dict, training_args):
        self.training_setup(training_args)
        self.densification_postfix(
            params_dict["xyz"],
            params_dict["features_dc"],
            params_dict["features_rest"],
            params_dict["opacity"],
            params_dict["scaling"],
            params_dict["rotation"])

    def build_covariance_from_scaling_rotation(self, scaling, scaling_modifier, rotation, device:int=0):
        L = build_scaling_rotation(scaling_modifier * scaling, rotation, deivce=device)
        actual_covariance = L @ L.transpose(1, 2)
        symm = strip_symmetric(actual_covariance, device=device)
        return symm

    def setup_functions(self) -> None:
        self.scaling_activation = torch.exp
        self.scaling_inverse_activation = torch.log
        self.opacity_activation = torch.sigmoid
        self.inverse_opacity_activation = inverse_sigmoid
        self.rotation_activation = torch.nn.functional.normalize

    def capture_dict(self, non_blocking=False, device:str="cpu") -> dict:
        dict = {
            "active_sh_degree": self.active_sh_degree,
            "xyz": self._xyz.clone().detach().to(device, non_blocking=non_blocking),
            "features_dc": self._features_dc.clone().detach().to(device, non_blocking=non_blocking),
            "features_rest": self._features_rest.clone().detach().to(device, non_blocking=non_blocking),
            "scaling": self._scaling.clone().detach().to(device, non_blocking=non_blocking),
            "rotation": self._rotation.clone().detach().to(device, non_blocking=non_blocking),
            "opacity": self._opacity.clone().detach().to(device, non_blocking=non_blocking),
            "max_radii2D": self.max_radii2D.clone().detach().to(device, non_blocking=non_blocking),
            "xyz_gradient_accum": self.xyz_gradient_accum.clone().detach().to(device, non_blocking=non_blocking),
            "denom": self.denom.clone().detach().to(device, non_blocking=non_blocking),
            "spatial_lr_scale": self.spatial_lr_scale,
            # "optimizer": self.optimizer.state_dict(),
        }
        return dict

    def get_size(self):
        return self._xyz.shape[0]

    def get_scaling(self):
        if self.isotropic:
            scale = self.scaling_activation(self._scaling)[:, 0:1]  # Extract the first column
            scales = scale.repeat(1, 3)  # Replicate this column three times
            return scales
        return self.scaling_activation(self._scaling)

    def get_rotation(self):
        return self.rotation_activation(self._rotation)

    def get_xyz(self):
        return self._xyz

    def get_features(self):
        features_dc = self._features_dc
        features_rest = self._features_rest
        return torch.cat((features_dc, features_rest), dim=1)

    def get_opacity(self):
        return self.opacity_activation(self._opacity)

    def get_active_sh_degree(self):
        return self.active_sh_degree

    def get_covariance(self, scaling_modifier=1, device:int=0):
        return self.build_covariance_from_scaling_rotation(self.get_scaling(), scaling_modifier, self._rotation, device=device)

    def add_points(self, pcd:o3d.geometry.PointCloud, global_scale_init=True):
        if self.VERBOSE:
            print(f"Number of add points: {len(pcd.points)}")

        fused_point_cloud = torch.tensor(np.asarray(pcd.points), device=self.device).float()
        fused_point_num = fused_point_cloud.shape[0]
        fused_color = RGB2SH(torch.tensor(np.asarray(pcd.colors), device=self.device).float())
        features = (torch.zeros((fused_color.shape[0], 3, (self.max_sh_degree + 1) ** 2), dtype=torch.float, device=self.device))
        features[:, :3, 0] = fused_color
        features[:, 3:, 1:] = 0.0

        if global_scale_init:
            global_points = torch.cat((self.get_xyz(),torch.from_numpy(np.asarray(pcd.points)).float().to(self.device)))
            dist2 = torch.clamp_min(distCUDA2(global_points), 0.0000001)
            dist2 = dist2[self.get_size():]
        else:
            dist2 = torch.clamp_min(distCUDA2(torch.from_numpy(np.asarray(pcd.points)).float().to(self.device)), 0.0000001)
        scales = torch.log(1.0 * torch.sqrt(dist2))[..., None].repeat(1, 3)
        # scales = torch.log(0.001 * torch.ones_like(dist2))[..., None].repeat(1, 3)

        rots = torch.zeros((fused_point_num, 4), device=self.device)
        rots[:, 0] = 1
        opacities = inverse_sigmoid(0.5 * torch.ones((fused_point_num, 1), dtype=torch.float, device=self.device))

        new_xyz = nn.Parameter(fused_point_cloud.requires_grad_(True))
        new_features_dc = nn.Parameter(features[:, :, 0:1].transpose(1, 2).contiguous().requires_grad_(True))
        new_features_rest = nn.Parameter(features[:, :, 1:].transpose(1, 2).contiguous().requires_grad_(True))
        new_scaling = nn.Parameter(scales.requires_grad_(True))
        new_rotation = nn.Parameter(rots.requires_grad_(True))
        new_opacities = nn.Parameter(opacities.requires_grad_(True))
        self.densification_postfix(
            new_xyz,
            new_features_dc,
            new_features_rest,
            new_opacities,
            new_scaling,
            new_rotation,
        )

    def add_points_with_edge(self, all_pts:np.ndarray, sample_ids:np.ndarray,
                             edge:torch.Tensor, depth:torch.Tensor, global_scale_init=True,
                             depth_thres:float = 0.025):
        """
        Add new points to GaussianModel
        Args:
            all_pts (np.ndarray, N*6): A point cloud of shape (N, 6) 
                with last dimension representing (x, y, z, r, g, b)
            sample_ids (np.ndarray, 1D): The indexs corresponding to the image 
                of the points in the pcd that needs to be added
            edge (torch.Tensor, dtype:bool): The edge image
            depth (torch.Tensor, dtype:float): The depth map
        """
        sample_pts_num = sample_ids.shape[0]

        all_pts:torch.Tensor = torch.from_numpy(all_pts).float().to(self.device, non_blocking=True)
        sample_ids:torch.Tensor = torch.from_numpy(sample_ids).to(self.device, non_blocking=True)

        # variable init
        edge_width = edge.shape[1]
        offsets = torch.tensor([-1, 0, 1], dtype=torch.int, device=self.device)

        torch.cuda.synchronize(self.device)

        # Get patches for the edge_sample_ids
        edge_sample_ids = sample_ids[edge.flatten()[sample_ids]]
        num_patches = edge_sample_ids.shape[0]
        rows = torch.div(edge_sample_ids, edge_width, rounding_mode="floor")
        cols = torch.remainder(edge_sample_ids, edge_width)
        rows_grid = rows[:, None, None] + offsets[None, :, None]
        cols_grid = cols[:, None, None] + offsets[None, None, :]
        patches_depth = torch.empty((num_patches, 3, 3), dtype=torch.float, device=self.device)
        patches_edge = torch.empty((num_patches, 3, 3), dtype=torch.bool, device=self.device)
        patches_depth = depth[rows_grid, cols_grid]
        patches_edge = edge[rows_grid, cols_grid]
        # Filter using depth 
        mid_elements = patches_depth[:, 1, 1].unsqueeze(1).unsqueeze(2)
        patches_edge = (torch.abs(patches_depth-mid_elements) < depth_thres) & patches_edge
        # Remove bad edge
        good_edge_cnts = patches_edge.sum(dim=(1,2))
        good_edge_mask = (1<good_edge_cnts) & (good_edge_cnts<4)

        # print(f"Good edge points: {good_edge_mask[good_edge_mask].shape[0]:d}/{num_patches:d}.")
        if good_edge_mask[good_edge_mask].shape[0]>0:
            patches_edge = patches_edge[good_edge_mask]
            rows, cols = rows[good_edge_mask], cols[good_edge_mask]
            # Get edge point pairs
            rows_grid = rows[:, None, None] + offsets[None, :, None]
            cols_grid = cols[:, None, None] + offsets[None, None, :]
            patches_edge_index = (edge_width * rows_grid + cols_grid) * patches_edge
            mid_elements = patches_edge_index[:, 1, 1].unsqueeze(1).unsqueeze(2)
            valid_mask = patches_edge & (patches_edge_index != mid_elements)
            middle_indices_expanded = mid_elements.expand_as(patches_edge_index)
            pairs = torch.stack((middle_indices_expanded[valid_mask], patches_edge_index[valid_mask]), dim=1)
            pairs, _ = torch.sort(pairs, dim=1)
            pairs:torch.Tensor = torch.unique(pairs, dim=0)
            pair_points = all_pts[pairs, :3]
            p1 = pair_points[:, 0, :]
            p2 = pair_points[:, 1, :]
            vectors = p2 - p1
            edge_gaussian_num = vectors.shape[0]

            pairs_flat = torch.unique(pairs.flatten())
            # sample_ids = sample_ids[~torch.isin(sample_ids, pairs_flat)]
            sample_pts_num = sample_ids.shape[0]
            total_pts_num = sample_pts_num + edge_gaussian_num
            # print(f"Add points: {edge_gaussian_num:d}/{total_pts_num:d}")

            # xyz
            xyz = torch.empty(total_pts_num, 3, dtype=torch.float, device=self.device)
            xyz[:sample_pts_num, :] = all_pts[sample_ids, :3]
            xyz[sample_pts_num:, :] = (p1 + p2) / 2

            # color
            rgb = torch.empty(total_pts_num, 3, dtype=torch.float, device=self.device)
            rgb[:sample_pts_num, :] = all_pts[sample_ids, 3:]
            rgb[sample_pts_num:, :] = (all_pts[pairs[:,0], 3:] + all_pts[pairs[:,1], 3:]) / 2
            features = torch.zeros((total_pts_num, 3, (self.max_sh_degree+1)**2), dtype=torch.float, device=self.device)
            features[:, :3, 0] = RGB2SH(rgb / 255.0)
            features[:, 3:, 1:] = 0.0

            # scale
            if global_scale_init:
                global_points = torch.cat((self.get_xyz(), all_pts[sample_ids, :3]))
                dist2 = distCUDA2(global_points)
                dist2 = torch.clamp_min(dist2, 0.0000001)
                dist2 = dist2[self.get_size():]
            else:
                dist2 = torch.clamp_min(distCUDA2(all_pts[sample_ids, :3]), 0.0000001)
            distances = torch.sqrt(torch.sum(vectors ** 2, dim=1))
            edge_scales = torch.empty(edge_gaussian_num, 3, dtype=torch.float, device=self.device)
            edge_scales[:, 0] = 1.25 * distances
            edge_scales[:, 1] = edge_scales[:, 2] = 0.5 * distances
            scales = torch.empty(total_pts_num, 3, dtype=torch.float, device=self.device)
            scales[:sample_pts_num, :] = torch.log(1.0 * torch.sqrt(dist2))[..., None].repeat(1, 3)
            scales[sample_pts_num:, :] = torch.log(edge_scales)

            # rotation
            norm_vectors = vectors / vectors.norm(dim=1, keepdim=True, dtype=torch.float)
            x_axis = torch.tensor([1, 0, 0], dtype=torch.float, device=self.device)
            cross_prod = torch.cross(x_axis.expand_as(norm_vectors), norm_vectors)
            angles = torch.acos(torch.sum(x_axis * norm_vectors, dim=1))
            axis_norm = cross_prod.norm(dim=1, keepdim=True)
            axis = cross_prod / axis_norm
            axis[axis_norm.squeeze() == 0] = torch.tensor([1, 0, 0], dtype=torch.float, device=self.device)
            half_angles = angles / 2
            quaternions = torch.zeros((edge_gaussian_num, 4), dtype=torch.float, device=self.device)
            quaternions[:, 0] = torch.cos(half_angles)
            quaternions[:, 1:] = axis * torch.sin(half_angles).unsqueeze(1)
            rots = torch.zeros((total_pts_num, 4), dtype=torch.float, device=self.device)
            rots[:sample_pts_num, 0] = 1
            rots[sample_pts_num:, :] = quaternions

            # opacities
            opacities = torch.full((total_pts_num, 1), 0.5, dtype=torch.float, device=self.device)
            opacities[:sample_pts_num][torch.isin(sample_ids, pairs_flat)] = 0.1
            opacities = inverse_sigmoid(opacities)

        else:
            # color
            xyz = all_pts[sample_ids, :3]
            fused_color = RGB2SH(all_pts[sample_ids, 3:] / 255.0)
            features = (torch.zeros((sample_pts_num, 3, (self.max_sh_degree + 1) ** 2), dtype=torch.float, device=self.device))
            features[:, :3, 0] = fused_color
            features[:, 3:, 1:] = 0.0

            # scale
            if global_scale_init:
                global_points = torch.cat((self.get_xyz(), all_pts[sample_ids, :3]))
                dist2 = torch.clamp_min(distCUDA2(global_points), 0.0000001)
                dist2 = dist2[self.get_size():] # # Get distance between points
            else:
                dist2 = torch.clamp_min(distCUDA2(all_pts[sample_ids, :3]), 0.0000001)
            scales = torch.log(1.0 * torch.sqrt(dist2))[..., None].repeat(1, 3) # scales.shape = [N,3]
            # scales = torch.log(0.001 * torch.ones_like(dist2))[..., None].repeat(1, 3)

            # rotation
            rots = torch.zeros((sample_pts_num, 4), dtype=torch.float, device=self.device)
            rots[:, 0] = 1

            # opacities
            opacities = inverse_sigmoid(0.5 * torch.ones((sample_pts_num, 1), dtype=torch.float, device=self.device))

        new_xyz = nn.Parameter(xyz.requires_grad_(True))
        new_features_dc = nn.Parameter(features[:, :, 0:1].transpose(1, 2).contiguous().requires_grad_(True))
        new_features_rest = nn.Parameter(features[:, :, 1:].transpose(1, 2).contiguous().requires_grad_(True))
        new_scaling = nn.Parameter(scales.requires_grad_(True))
        new_rotation = nn.Parameter(rots.requires_grad_(True))
        new_opacities = nn.Parameter(opacities.requires_grad_(True))
        self.densification_postfix(
            new_xyz,
            new_features_dc,
            new_features_rest,
            new_opacities,
            new_scaling,
            new_rotation,
        )

    def update_learning_rate(self, iteration):
        ''' Learning rate scheduling per step '''
        for param_group in self.optimizer.param_groups:
            if param_group["name"] == "xyz":
                lr = self.xyz_scheduler_args(iteration)
                param_group['lr'] = lr
                return lr

    def training_setup(self, training_args, exposure_ab=None):
        self.percent_dense = training_args.percent_dense
        xyz_shape = self.get_xyz().shape[0]
        self.xyz_gradient_accum = torch.zeros((xyz_shape, 1), dtype=torch.float, device=self.device)
        self.denom = torch.zeros((xyz_shape, 1), dtype=torch.float, device=self.device)

        params = [
            {"params": [self._xyz], "lr": training_args.position_lr_init, "name": "xyz"},
            {"params": [self._features_dc], "lr": training_args.feature_lr, "name": "f_dc"},
            {"params": [self._features_rest], "lr": training_args.feature_lr / 20.0, "name": "f_rest"},
            {"params": [self._opacity], "lr": training_args.opacity_lr, "name": "opacity"},
            {"params": [self._scaling], "lr": training_args.scaling_lr, "name": "scaling"},
            {"params": [self._rotation], "lr": training_args.rotation_lr, "name": "rotation"},
        ]

        if exposure_ab is not None:
            params.extend([
                {"params": [exposure_ab[0]], "lr": 0.01, "name": "exposure_a"},
                {"params": [exposure_ab[1]], "lr": 0.01, "name": "exposure_b"}]
            )

        self.optimizer = torch.optim.Adam(params, lr=0.0, eps=1e-15)
        self.xyz_scheduler_args = get_expon_lr_func(
            lr_init=training_args.position_lr_init * self.spatial_lr_scale,
            lr_final=training_args.position_lr_final * self.spatial_lr_scale,
            lr_delay_mult=training_args.position_lr_delay_mult,
            max_steps=training_args.position_lr_max_steps,
        )

    def training_setup_camera(self, cam_rot, cam_trans, cfg, exposure_ab=None):
        self.xyz_gradient_accum = torch.zeros(
            (self.get_xyz().shape[0], 1), device=self.device
        )
        self.denom = torch.zeros((self.get_xyz().shape[0], 1), device=self.device)

        params = [
            {"params": [cam_rot], "lr": cfg["cam_rot_lr"],
                "name": "cam_unnorm_rot"},
            {"params": [cam_trans], "lr": cfg["cam_trans_lr"],
                "name": "cam_trans"},
        ]
        if exposure_ab is not None:
            params.extend([
                {"params": [exposure_ab[0]], "lr": 0.01, "name": "exposure_a"},
                {"params": [exposure_ab[1]], "lr": 0.01, "name": "exposure_b"}]
            )
        self.optimizer = torch.optim.Adam(params, amsgrad=True)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, "min",
            factor=cfg["scheduler_factor"], patience=cfg["scheduler_patience"], verbose=False)

    def construct_list_of_attributes(self):
        l = ["x", "y", "z", "nx", "ny", "nz"]
        # All channels except the 3 DC
        for i in range(self._features_dc.shape[1] * self._features_dc.shape[2]):
            l.append("f_dc_{}".format(i))
        for i in range(self._features_rest.shape[1] * self._features_rest.shape[2]):
            l.append("f_rest_{}".format(i))
        l.append("opacity")
        for i in range(self._scaling.shape[1]):
            l.append("scale_{}".format(i))
        for i in range(self._rotation.shape[1]):
            l.append("rot_{}".format(i))
        return l

    def save_ply(self, path):
        Path(path).parent.mkdir(parents=True, exist_ok=True)

        xyz = self._xyz.detach().cpu().numpy()
        normals = np.zeros_like(xyz)
        f_dc = (
            self._features_dc.detach()
            .transpose(1, 2)
            .flatten(start_dim=1)
            .contiguous()
            .cpu()
            .numpy())
        f_rest = (
            self._features_rest.detach()
            .transpose(1, 2)
            .flatten(start_dim=1)
            .contiguous()
            .cpu()
            .numpy())
        opacities = self._opacity.detach().cpu().numpy()
        if self.isotropic:
            # tile into shape (P, 3)
            scale = np.tile(self._scaling.detach().cpu().numpy()[:, 0].reshape(-1, 1), (1, 3))
        else:
            scale = self._scaling.detach().cpu().numpy()
        rotation = self._rotation.detach().cpu().numpy()

        dtype_full = [(attribute, "f4") for attribute in self.construct_list_of_attributes()]

        elements = np.empty(xyz.shape[0], dtype=dtype_full)
        attributes = np.concatenate((xyz, normals, f_dc, f_rest, opacities, scale, rotation), axis=1)
        elements[:] = list(map(tuple, attributes))
        el = PlyElement.describe(elements, "vertex")
        PlyData([el]).write(path)

    def load_ply(self, path):
        plydata = PlyData.read(path)

        xyz = np.stack((
                np.asarray(plydata.elements[0]["x"]),
                np.asarray(plydata.elements[0]["y"]),
                np.asarray(plydata.elements[0]["z"])),
                axis=1)
        opacities = np.asarray(plydata.elements[0]["opacity"])[..., np.newaxis]

        features_dc = np.zeros((xyz.shape[0], 3, 1))
        features_dc[:, 0, 0] = np.asarray(plydata.elements[0]["f_dc_0"])
        features_dc[:, 1, 0] = np.asarray(plydata.elements[0]["f_dc_1"])
        features_dc[:, 2, 0] = np.asarray(plydata.elements[0]["f_dc_2"])

        extra_f_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("f_rest_")]
        extra_f_names = sorted(extra_f_names, key=lambda x: int(x.split("_")[-1]))
        assert len(extra_f_names) == 3 * (self.max_sh_degree + 1) ** 2 - 3
        features_extra = np.zeros((xyz.shape[0], len(extra_f_names)))
        for idx, attr_name in enumerate(extra_f_names):
            features_extra[:, idx] = np.asarray(plydata.elements[0][attr_name])
        # Reshape (P,F*SH_coeffs) to (P, F, SH_coeffs except DC)
        features_extra = features_extra.reshape((features_extra.shape[0], 3, (self.max_sh_degree + 1) ** 2 - 1))

        scale_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("scale_")]
        scale_names = sorted(scale_names, key=lambda x: int(x.split("_")[-1]))
        scales = np.zeros((xyz.shape[0], len(scale_names)))
        for idx, attr_name in enumerate(scale_names):
            scales[:, idx] = np.asarray(plydata.elements[0][attr_name])

        rot_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("rot")]
        rot_names = sorted(rot_names, key=lambda x: int(x.split("_")[-1]))
        rots = np.zeros((xyz.shape[0], len(rot_names)))
        for idx, attr_name in enumerate(rot_names):
            rots[:, idx] = np.asarray(plydata.elements[0][attr_name])

        self._xyz = nn.Parameter(torch.tensor(xyz, dtype=torch.float, device=self.device).requires_grad_(True))
        self._features_dc = nn.Parameter(
            torch.tensor(features_dc, dtype=torch.float, device=self.device)
            .transpose(1, 2).contiguous().requires_grad_(True))
        self._features_rest = nn.Parameter(
            torch.tensor(features_extra, dtype=torch.float, device=self.device)
            .transpose(1, 2)
            .contiguous()
            .requires_grad_(True)
        )
        self._opacity = nn.Parameter(torch.tensor(opacities, dtype=torch.float, device=self.device).requires_grad_(True))
        self._scaling = nn.Parameter(torch.tensor(scales, dtype=torch.float, device=self.device).requires_grad_(True))
        self._rotation = nn.Parameter(torch.tensor(rots, dtype=torch.float, device=self.device).requires_grad_(True))

        self.active_sh_degree = self.max_sh_degree

    def replace_tensor_to_optimizer(self, tensor, name):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            if group["name"] == name:
                stored_state = self.optimizer.state.get(group["params"][0], None)
                stored_state["exp_avg"] = torch.zeros_like(tensor, device=self.device)
                stored_state["exp_avg_sq"] = torch.zeros_like(tensor, device=self.device)

                del self.optimizer.state[group["params"][0]]
                group["params"][0] = nn.Parameter(tensor.requires_grad_(True))
                self.optimizer.state[group["params"][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors

    def _prune_optimizer(self, mask):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            if "exposure" not in group["name"]:
                stored_state = self.optimizer.state.get(group["params"][0], None)
                if stored_state is not None:
                    stored_state["exp_avg"] = stored_state["exp_avg"][mask]
                    stored_state["exp_avg_sq"] = stored_state["exp_avg_sq"][mask]

                    del self.optimizer.state[group["params"][0]]
                    group["params"][0] = nn.Parameter((group["params"][0][mask].requires_grad_(True)))
                    self.optimizer.state[group["params"][0]] = stored_state
                    optimizable_tensors[group["name"]] = group["params"][0]
                else:
                    group["params"][0] = nn.Parameter(group["params"][0][mask].requires_grad_(True))
                    optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors

    def prune_points(self, mask):
        valid_points_mask = ~mask
        optimizable_tensors = self._prune_optimizer(valid_points_mask)

        self._xyz = optimizable_tensors["xyz"]
        self._features_dc = optimizable_tensors["f_dc"]
        self._features_rest = optimizable_tensors["f_rest"]
        self._opacity = optimizable_tensors["opacity"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]

        self.xyz_gradient_accum = self.xyz_gradient_accum[valid_points_mask]

        self.denom = self.denom[valid_points_mask]
        self.max_radii2D = self.max_radii2D[valid_points_mask]

    def cat_tensors_to_optimizer(self, tensors_dict):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            assert len(group["params"]) == 1
            if group["name"] in tensors_dict:
                extension_tensor = tensors_dict[group["name"]]
                stored_state = self.optimizer.state.get(group["params"][0], None)
                if stored_state is not None:
                    stored_state["exp_avg"] = torch.cat(
                        (stored_state["exp_avg"], torch.zeros_like(extension_tensor, device=self.device)), dim=0)
                    stored_state["exp_avg_sq"] = torch.cat(
                        (stored_state["exp_avg_sq"], torch.zeros_like(extension_tensor, device=self.device)), dim=0)

                    del self.optimizer.state[group["params"][0]]
                    group["params"][0] = nn.Parameter(
                        torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
                    self.optimizer.state[group["params"][0]] = stored_state

                    optimizable_tensors[group["name"]] = group["params"][0]
                else:
                    group["params"][0] = nn.Parameter(
                        torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
                    optimizable_tensors[group["name"]] = group["params"][0]

        return optimizable_tensors

    def densification_postfix(self, new_xyz, new_features_dc, new_features_rest,
                              new_opacities, new_scaling, new_rotation):
        d = {
            "xyz": new_xyz,
            "f_dc": new_features_dc,
            "f_rest": new_features_rest,
            "opacity": new_opacities,
            "scaling": new_scaling,
            "rotation": new_rotation,
        }

        optimizable_tensors = self.cat_tensors_to_optimizer(d)
        self._xyz = optimizable_tensors["xyz"]
        self._features_dc = optimizable_tensors["f_dc"]
        self._features_rest = optimizable_tensors["f_rest"]
        self._opacity = optimizable_tensors["opacity"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]

        self.xyz_gradient_accum = torch.zeros((self.get_xyz().shape[0], 1), device=self.device)
        self.denom = torch.zeros((self.get_xyz().shape[0], 1), device=self.device)
        self.max_radii2D = torch.zeros(
            (self.get_xyz().shape[0]), device=self.device)

    def add_densification_stats(self, viewspace_point_tensor, update_filter):
        self.xyz_gradient_accum[update_filter] += torch.norm(
            viewspace_point_tensor.grad[update_filter, :2], dim=-1, keepdim=True)
        self.denom[update_filter] += 1

    def oneupSHdegree(self):
        if self.active_sh_degree < self.max_sh_degree:
            self.active_sh_degree += 1

    def create_from_pcd(self, pcd: BasicPointCloud, spatial_lr_scale: float):
        self.spatial_lr_scale = spatial_lr_scale
        if type(pcd) is BasicPointCloud:
            fused_point_cloud = torch.tensor(
                np.asarray(pcd.points), dtype=torch.float, device=self.device)
            fused_color = RGB2SH(torch.tensor(
                np.asarray(pcd.colors), dtype=torch.float, device=self.device))
        else:
            fused_point_cloud = torch.tensor(
                np.asarray(pcd._xyz), dtype=torch.float, device=self.device)
            fused_color = RGB2SH(torch.tensor(
                np.asarray(pcd._rgb), dtype=torch.float, device=self.device))
        features = torch.zeros(
            (fused_color.shape[0], 3, (self.max_sh_degree + 1) ** 2), dtype=torch.float, device=self.device)
        features[:, :3, 0] = fused_color
        features[:, 3:, 1:] = 0.0

        fused_point_num = fused_point_cloud.shape[0]
        if self.VERBOSE:
            print(f"Number of points at initialisation : {fused_point_num:d}")

        dist2 = torch.clamp_min(
            distCUDA2(fused_point_cloud.detach().clone().float().to(device=self.device)), 0.0000001)
        scales = torch.log(torch.sqrt(dist2))[..., None].repeat(1, 3)
        rots = torch.zeros((fused_point_num, 4), device=self.device)
        rots[:, 0] = 1

        opacities = inverse_sigmoid(
            0.1 * torch.ones((fused_point_num, 1), dtype=torch.float, device=self.device))

        self._xyz = nn.Parameter(fused_point_cloud.requires_grad_(True))
        self._features_dc = nn.Parameter(features[:, :, 0:1].transpose(
            1, 2).contiguous().requires_grad_(True))
        self._features_rest = nn.Parameter(
            features[:, :, 1:].transpose(1, 2).contiguous().requires_grad_(True))
        self._scaling = nn.Parameter(scales.requires_grad_(True))
        self._rotation = nn.Parameter(rots.requires_grad_(True))
        self._opacity = nn.Parameter(opacities.requires_grad_(True))
        self.max_radii2D = torch.zeros((self.get_xyz().shape[0]), device=self.device)
