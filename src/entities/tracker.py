from argparse import ArgumentParser

import os
import numpy as np
import torch
import torch.nn.functional as F
import torchvision
from scipy.spatial.transform import Rotation as R
import concurrent.futures

from src.entities.arguments import OptimizationParams
from src.entities.losses import l1_loss
from src.entities.gaussian_model import GaussianModel
from src.entities.logger import Logger
from src.entities.datasets import BaseDataset
from src.utils.gaussian_model_utils import build_rotation
from src.utils.tracker_utils import (compute_camera_opt_params,
                                     extrapolate_poses, multiply_quaternions,
                                     transformation_to_quaternion)
from src.utils.utils import (get_render_settings, np2torch,
                             render_gaussian_model, torch2np)

from VO.build.lib import VisualOdom

class Tracker(object):
    def __init__(self, config: dict, dataset: BaseDataset, logger: Logger, device:int=0) -> None:
        """ Initializes the Tracker with a given configuration, dataset, and logger.
        Args:
            config: Configuration dictionary specifying hyperparameters and operational settings.
            dataset: The dataset object providing access to the sequence of frames.
            logger: Logger object for logging the tracking process.
        """
        self.device = device
        self.dataset = dataset
        self.logger = logger
        self.config = config
        self.filter_alpha = self.config["tracking"]["filter_alpha"]
        self.filter_outlier_depth = self.config["tracking"]["filter_outlier_depth"]
        self.alpha_thre = self.config["tracking"]["alpha_thre"]
        self.soft_alpha = self.config["tracking"]["soft_alpha"]
        self.mask_invalid_depth_in_color_loss = self.config["tracking"]["mask_invalid_depth"]
        self.w_color_loss = self.config["tracking"]["w_color_loss"]
        self.transform = torchvision.transforms.ToTensor()
        self.opt = OptimizationParams(ArgumentParser(description="Training script parameters"))
        self.frame_depth_loss = []
        self.frame_color_loss = []
        self.odometry_type = self.config["tracking"]["odometry_type"]
        self.help_camera_initialization = self.config["tracking"]["help_camera_initialization"]
        self.init_err_ratio = self.config["tracking"]["init_err_ratio"]
        self.enable_exposure = self.config["tracking"]["enable_exposure"]
        self.NUM_ITERS = self.config["tracking"]["iterations"]
        self.early_stop_thre:float = self.config["tracking"]["early_stop_thre"]
        self.early_stop_cnt:int = self.config["tracking"]["early_stop_cnt"]

        # For debug. Statistical information on the number of iterations
        self.init_pose_cnt = {"const_speed":0, "previous":0}
        if self.config["verbose"]:
            self.iter_cnt = []
            self.iter_cnt_min_loss = []

        if self.odometry_type == "odometer" or self.help_camera_initialization:
            self.vo = VisualOdom.VisualOdom(
                os.path.join(os.path.dirname(__file__), "../../configs/VO", config["tracking"]["vo_setting_file"]),
                os.path.join(os.path.dirname(__file__), "../../configs/VO", self.config["dataset_name"], 
                             self.config["data"]["scene_name"]+".yaml")
            )
            self.init_pose_cnt["odometer"] = 0

    def compute_losses(self, gaussian_model: GaussianModel, render_settings: dict,
                       opt_cam_rot: torch.Tensor, opt_cam_trans: torch.Tensor,
                       gt_color: torch.Tensor, gt_depth: torch.Tensor, depth_mask: torch.Tensor,
                       exposure_ab=None) -> tuple:
        """ Computes the tracking losses with respect to ground truth color and depth.
        Args:
            gaussian_model: The current state of the Gaussian model of the scene.
            render_settings: Dictionary containing rendering settings such as image dimensions and camera intrinsics.
            opt_cam_rot: Optimizable tensor representing the camera's rotation.
            opt_cam_trans: Optimizable tensor representing the camera's translation.
            gt_color: Ground truth color image tensor.
            gt_depth: Ground truth depth image tensor.
            depth_mask: Binary mask indicating valid depth values in the ground truth depth image.
        Returns:
            A tuple containing losses and renders
        """
        rel_transform = torch.eye(4, dtype=torch.float, device=self.device)
        rel_transform[:3, :3] = build_rotation(F.normalize(opt_cam_rot[None]), device=self.device)[0]
        rel_transform[:3, 3] = opt_cam_trans

        pts = gaussian_model.get_xyz()
        pts_ones = torch.ones(pts.shape[0], 1, dtype=torch.float, device=self.device)
        pts4 = torch.cat((pts, pts_ones), dim=1)
        transformed_pts = (rel_transform @ pts4.T).T[:, :3]

        quat = F.normalize(opt_cam_rot[None])
        _rotations = multiply_quaternions(gaussian_model.get_rotation(), quat.unsqueeze(0)).squeeze(0)

        render_dict = render_gaussian_model(gaussian_model, render_settings,
                                            override_means_3d=transformed_pts, override_rotations=_rotations)
        rendered_color, rendered_depth = render_dict["color"], render_dict["depth"]
        if self.enable_exposure:
            rendered_color = torch.clamp(torch.exp(exposure_ab[0]) * rendered_color + exposure_ab[1], 0, 1.)
        alpha_mask = render_dict["alpha"] > self.alpha_thre

        tracking_mask = torch.ones_like(alpha_mask).bool()
        tracking_mask &= depth_mask
        if self.filter_alpha:
            tracking_mask &= alpha_mask
        if self.filter_outlier_depth:
            depth_err = torch.abs(rendered_depth - gt_depth) * depth_mask
            if torch.median(depth_err) > 0:
                tracking_mask &= depth_err < 50 * torch.median(depth_err)

        color_loss = l1_loss(rendered_color, gt_color, agg="none") + 1e-8
        depth_loss = (l1_loss(rendered_depth, gt_depth, agg="none")  + 1e-8) * tracking_mask

        if self.soft_alpha:
            alpha = render_dict["alpha"] ** 3
            color_loss *= alpha
            depth_loss *= alpha
            if self.mask_invalid_depth_in_color_loss:
                color_loss *= tracking_mask
        else:
            color_loss *= tracking_mask

        color_loss = color_loss.sum() / torch.sum(color_loss>0)
        depth_loss = depth_loss.sum() / torch.sum(depth_loss>0)

        return color_loss, depth_loss, rendered_color, rendered_depth, alpha_mask

    def init_pose_min_loss(self, gaussian_model:GaussianModel, render_settings: dict, init_c2ws:dict,
                           gt_color: torch.Tensor, gt_depth: torch.Tensor, depth_mask: torch.Tensor,
                           exposure_ab, vo_future:concurrent.futures.Future) -> tuple[np.ndarray, float, float]:
        """ Find the min loss of init pose
        Args:
            gaussian_model: The current state of the Gaussian model of the scene.
            render_settings: Dictionary containing rendering settings such as image dimensions and camera intrinsics.
            init_c2ws: Dictionary containing init pose method name:str, init pose:ndarray.
            gt_color: Ground truth color image:tensor.
            gt_depth: Ground truth depth image:tensor.
            depth_mask: Binary mask indicating valid depth values in the ground truth depth image:tensor.
            vo_future: The future object of visual odometry.
        Returns:
            tuple: init_c2w:ndarray, min_color_loss:float, min_depth_loss:float
        """
        with torch.no_grad():
            min_loss = float("inf")
            last_w2c = np.linalg.inv(init_c2ws["previous"])
            for name, c2w in init_c2ws.items():
                if name == "odometer":
                    c2w = vo_future.result()
                opt_cam_rot, opt_cam_trans = compute_camera_opt_params(np.linalg.inv(c2w @ last_w2c), device=self.device)
                gaussian_model.training_setup_camera(opt_cam_rot, opt_cam_trans, self.config["tracking"], exposure_ab)
                color_loss, depth_loss, _, _, _ = self.compute_losses(gaussian_model, render_settings, 
                    opt_cam_rot, opt_cam_trans, gt_color, gt_depth, depth_mask, exposure_ab)
                total_loss = (self.w_color_loss * color_loss + (1 - self.w_color_loss) * depth_loss)
                total_loss_float = total_loss.item()
                if self.config["verbose"]:
                    print(f"Init pose method: {name}, loss:{total_loss_float:.6f}")
                if total_loss_float < min_loss:
                    min_color_loss = color_loss.item()
                    min_depth_loss = depth_loss.item()
                    min_loss = total_loss_float
                    min_name = name
                    init_c2w = c2w

            self.init_pose_cnt[min_name] = self.init_pose_cnt[min_name] + 1
            if self.config["verbose"]:
                print(f"Using {min_name} init pose\n")   
        return init_c2w, min_color_loss, min_depth_loss

    def report(self):
        if self.config["verbose"]:
            print(f"\nInit pose cnt report:")
            for name, cnt in self.init_pose_cnt.items():
                print(f"{name}: {cnt}")
            print(f"Iteration count avg:{np.average(self.iter_cnt)}, " +
                f"min:{min(self.iter_cnt)}, max:{max(self.iter_cnt)}.")
            print(f"Iteration to min loss avg:{np.average(self.iter_cnt_min_loss)}, " +
                f"min:{min(self.iter_cnt_min_loss)}, max:{max(self.iter_cnt_min_loss)}.")

    def track(self, frame_id: int, gaussian_model: GaussianModel, prev_c2ws: np.ndarray) -> np.ndarray:
        """
        Updates the camera pose estimation for the current frame based on the provided image and depth, using either ground truth poses,
        constant speed assumption, or visual odometry.
        Args:
            frame_id: Index of the current frame being processed.
            gaussian_model: The current Gaussian model of the scene.
            prev_c2ws: Array containing the camera-to-world transformation matrices for the frames (i-2, i-1)
        Returns:
            The updated camera-to-world transformation matrix for the current frame.
        """
        if self.config["verbose"]:
            print(f"\nTracking frame {frame_id}")

        _, image, depth, gt_T_world_cam = self.dataset[frame_id]
        if self.odometry_type == "gt":
            return gt_T_world_cam
        vo_future = None
        if self.odometry_type == "odometer" or self.help_camera_initialization:
            image_origin, depth_origin = self.dataset.get_origin_image(frame_id)
            vo_future = concurrent.futures.ThreadPoolExecutor(max_workers=1).submit(
                self.vo.step, image_origin, depth_origin, self.dataset.timestamps[frame_id])

        last_c2w = prev_c2ws[-1]
        last_w2c = np.linalg.inv(last_c2w)

        render_settings = get_render_settings(self.dataset.width, self.dataset.height,
                                              self.dataset.intrinsics, last_w2c, device=self.device)
        gt_color = self.transform(image).to(self.device)
        gt_depth = np2torch(depth, self.device)
        depth_mask = gt_depth > 0.0
        if self.enable_exposure:
            exposure_ab = torch.nn.Parameter(torch.tensor(0.0, device=self.device)), \
                torch.nn.Parameter(torch.tensor(0.0, device=self.device))
        else:
            exposure_ab = None

        init_c2ws = {}
        init_c2ws["const_speed"] = extrapolate_poses(prev_c2ws)
        init_c2ws["previous"] = prev_c2ws[-1]
        if (self.odometry_type == "odometer" or self.help_camera_initialization) and frame_id >= 3:
            init_c2ws["odometer"] = None
        init_c2w, init_color_loss, init_depth_loss = self.init_pose_min_loss(gaussian_model, render_settings, init_c2ws,
                                gt_color, gt_depth, depth_mask, exposure_ab, vo_future)
        init_2_last = init_c2w @ last_w2c
        last_2_init = np.linalg.inv(init_2_last)
        gt_trans = np2torch(gt_T_world_cam[:3, 3])
        gt_quat = np2torch(R.from_matrix(gt_T_world_cam[:3, :3]).as_quat(canonical=True)[[3, 0, 1, 2]])

        num_iters = self.NUM_ITERS
        if len(self.frame_color_loss) > 0 and (
                init_color_loss > self.init_err_ratio * np.median(self.frame_color_loss) or 
                init_depth_loss > self.init_err_ratio * np.median(self.frame_depth_loss)):
            num_iters *= 2
            if self.config["verbose"]:
                print(f"Higher initial loss, increasing num_iters to {num_iters}")
            if self.help_camera_initialization and self.odometry_type != "odometer":
                if self.config["verbose"]:
                    print(f"re-init with odometer for frame {frame_id}")
                init_c2w = self.vo.getTwc(frame_id)
                init_2_last = init_c2w @ np.linalg.inv(last_c2w)
                last_2_init = np.linalg.inv(init_2_last)

        # Iteration counter
        if self.config["verbose"]:
            iter_cnt_min_loss = int(-1)
            iter_cnt = int(0)
        min_loss = float("inf")
        # Early stop
        prev_loss = float("inf")
        break_cnt = int(0)
        break_flag = False

        opt_cam_rot, opt_cam_trans = compute_camera_opt_params(last_2_init, device=self.device)
        gaussian_model.training_setup_camera(opt_cam_rot, opt_cam_trans, self.config["tracking"], exposure_ab)
        
        # Track main iteration
        for iter in range(num_iters):
            gaussian_model.optimizer.zero_grad(set_to_none=True)
            color_loss, depth_loss, _, _, _, = self.compute_losses(gaussian_model, render_settings,
                            opt_cam_rot, opt_cam_trans, gt_color, gt_depth, depth_mask, exposure_ab)
            total_loss = (self.w_color_loss * color_loss + (1 - self.w_color_loss) * depth_loss)

            # Early stop
            with torch.no_grad():
                color_loss_float = color_loss.item()
                depth_loss_float = depth_loss.item()
                total_loss_float = total_loss.item()
                if self.config["verbose"]:
                    iter_cnt += 1
                if abs(total_loss_float-prev_loss) < self.early_stop_thre:
                    break_cnt += 1
                    if break_cnt > self.early_stop_cnt:
                        break_flag = True
                else:
                    break_cnt = 0
                prev_loss = total_loss_float

            if not break_flag:
                total_loss.backward()
                gaussian_model.optimizer.step()
                gaussian_model.scheduler.step(total_loss)

            with torch.no_grad():
                if total_loss_float < min_loss:
                    min_loss = total_loss_float
                    best_color_loss = color_loss_float
                    best_depth_loss = depth_loss_float
                    if self.config["verbose"]:
                        iter_cnt_min_loss = iter
                        best_last2cur = torch.eye(4, dtype=torch.float, device="cpu")
                        best_last2cur[:3, :3] = build_rotation(F.normalize(opt_cam_rot[None].clone().detach().cpu()), device=self.device)[0]
                        best_last2cur[:3, 3] = opt_cam_trans.clone().detach().cpu()
                    else:
                        best_cam_rot = opt_cam_rot.clone().detach()
                        best_cam_tran = opt_cam_trans.clone().detach()
                    break_cnt = 0

                if self.config["verbose"]:
                    if iter == num_iters - 1 or break_flag:
                        cur_w2c = torch.from_numpy(last_w2c) @ best_last2cur
                    else:
                        cur_quat = F.normalize(opt_cam_rot[None].clone().detach())
                        cur_rel_w2c = torch.eye(4)
                        cur_rel_w2c[:3, :3] = build_rotation(cur_quat, device=self.device)[0]
                        cur_rel_w2c[:3, 3] = opt_cam_trans.clone().detach()
                        cur_w2c = torch.from_numpy(last_w2c) @ cur_rel_w2c
                    cur_cam = transformation_to_quaternion(torch.inverse(cur_w2c))

                    # Log
                    if (gt_quat * cur_cam[:4]).sum() < 0:
                        gt_quat *= -1
                    if iter == num_iters - 1 or break_flag:
                        self.logger.log_tracking_iteration(frame_id, cur_cam, gt_quat, gt_trans,
                            total_loss_float, color_loss_float, depth_loss_float,
                            num_iters-1, num_iters, gaussian_model.optimizer.param_groups[0]["lr"],
                            wandb_output=True, print_output=True)
                    elif iter % 10 == 0:
                        self.logger.log_tracking_iteration(frame_id, cur_cam, gt_quat, gt_trans,
                            total_loss_float, color_loss_float, depth_loss_float,
                            iter, num_iters, gaussian_model.optimizer.param_groups[0]["lr"],
                            wandb_output=False, print_output=True)

                if break_flag:
                    if self.config["verbose"]:
                        print(f"Early exit at iter: {iter}.")
                    break

        self.frame_color_loss.append(best_color_loss)
        self.frame_depth_loss.append(best_depth_loss)

        # Count the number of iterations
        if self.config["verbose"]:
            self.iter_cnt.append(iter_cnt)
            self.iter_cnt_min_loss.append(iter_cnt_min_loss)
            print(f"Min loss:{min_loss:.8f} at iter:{iter_cnt_min_loss}.", flush=True)
        else:
            best_last2cur = torch.eye(4, dtype=torch.float, device=self.device)
            best_last2cur[:3, :3] = build_rotation(F.normalize(best_cam_rot[None]), device=self.device)[0]
            best_last2cur[:3, 3] = best_cam_tran
            best_last2cur = best_last2cur.cpu()

        final_c2w = torch.inverse(torch.from_numpy(last_w2c) @ best_last2cur)
        final_c2w[-1, :] = torch.tensor([0., 0., 0., 1.], dtype=final_c2w.dtype, device=final_c2w.device)
        if self.help_camera_initialization or self.odometry_type == "odometer":
            self.vo.setTwc(frame_id, torch2np(final_c2w))

        return torch2np(final_c2w), exposure_ab
