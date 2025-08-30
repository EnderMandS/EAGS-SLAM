""" This module includes the Logger class, which is responsible for logging for both Mapper and the Tracker """
from pathlib import Path
from typing import Union

import matplotlib.pyplot as plt
import numpy as np
import torch
import wandb

class Logger(object):

    def __init__(self, output_path: Union[Path, str], use_wandb=False, verbose:bool=False) -> None:
        self.VERBOSE:bool = verbose
        self.output_path = Path(output_path)
        (self.output_path / "mapping_vis").mkdir(exist_ok=True, parents=True)
        (self.output_path / "render_vis").mkdir(exist_ok=True, parents=True)
        self.use_wandb = use_wandb

        # Track
        self.pre_gt_t = None
        self.pre_gt_q = None
        self.pre_pose = None

    def log_tracking_iteration(self, frame_id:int, cur_pose, cur_gt_q, cur_gt_t, total_loss:float,
                               color_loss:float, depth_loss:float, iter:int, num_iters:int, lr:float,
                               wandb_output=False, print_output=False) -> None:
        """ Logs tracking iteration metrics including pose error, losses, and optionally reports to Weights & Biases.
        Logs the error between the current pose estimate and ground truth quaternion and translation,
        as well as various loss metrics. Can output to wandb if enabled and specified, and print to console.
        Args:
            frame_id: Identifier for the current frame.
            cur_pose: The current estimated pose as a tensor (quaternion + translation).
            cur_gt_q: Current ground truth quaternion.
            cur_gt_t: Current round truth translation.
            total_loss: Total computed loss for the current iteration.
            color_loss: Computed color loss for the current iteration.
            depth_loss: Computed depth loss for the current iteration.
            iter: The current iteration number.
            num_iters: The total number of iterations planned.
            lr: Current learn rate, echo multi 1e4.
            wandb_output: Whether to output the log to wandb.
            print_output: Whether to print the log output.
        """
        if not self.VERBOSE:
            return

        if self.pre_gt_t==None:
            self.pre_gt_t = cur_gt_t
            self.pre_gt_q = cur_gt_q
            self.pre_pose = cur_pose
            abs_q_err = abs_t_err = rel_q_err = rel_t_err = float(0)
        else:
            cur_q = cur_pose[:4]
            cur_t = cur_pose[4:]
            pre_q = self.pre_pose[:4]
            pre_t = self.pre_pose[4:]

            d_gt_q = torch.abs(self.pre_gt_q - cur_gt_q).mean().item()
            d_gt_t = torch.dist(self.pre_gt_t, cur_gt_t).item()
            d_rel_q = torch.abs(pre_q - cur_q).mean().item()
            d_rel_t = torch.dist(pre_t, cur_t).item()

            abs_q_err = torch.abs(cur_q - cur_gt_q).mean().item()
            abs_t_err = torch.dist(cur_t, cur_gt_t).item()
            rel_q_err = abs(d_gt_q - d_rel_q)
            rel_t_err = abs(d_gt_t - d_rel_t)

        if self.use_wandb and wandb_output:
            wandb.log({
                "Tracking/idx": frame_id,
                "Tracking/abs_q_err": abs_q_err,
                "Tracking/abs_t_err": abs_t_err,
                "Tracking/q_rel_err": rel_q_err,
                "Tracking/t_rel_err": rel_t_err,
                "Tracking/total_loss": total_loss,
                "Tracking/color_loss": color_loss,
                "Tracking/depth_loss": depth_loss,
                "Tracking/num_iters": num_iters,
            })

        if iter == num_iters - 1:
            self.pre_gt_t = cur_gt_t
            self.pre_gt_q = cur_gt_q
            self.pre_pose = cur_pose
            msg = f"frame: {frame_id}, Q_err abs: {abs_q_err:.6f} rel: {rel_q_err:.6f}, " + \
                  f"T_err abs: {abs_t_err:.6f} rel: {rel_t_err:.6f}"
        else:
            msg = f"iter:{iter:3d}, lr:{lr:.6f}, loss: t:{total_loss:.6f} " + \
                  f"c:{color_loss:.6f}, d:{depth_loss:.6f} " + \
                  f"T_e abs: {abs_t_err:.6f} rel: {rel_t_err:.6f}"

        if print_output:
            print(msg, flush=True)

    def log_mapping_iteration(self, frame_id, new_pts_num, model_size, iter_opt_time, opt_dict: dict) -> None:
        """ Logs mapping iteration metrics including the number of new points, model size, and optimization times,
        and optionally reports to Weights & Biases (wandb).
        Args:
            frame_id: Identifier for the current frame.
            new_pts_num: The number of new points added in the current mapping iteration.
            model_size: The total size of the model after the current mapping iteration.
            iter_opt_time: Time taken per optimization iteration.
            opt_dict: A dictionary containing optimization metrics such as PSNR, color loss, and depth loss.
        """
        if not self.VERBOSE:
            return
        if self.use_wandb:
            wandb.log({"Mapping/idx": frame_id,
                       "Mapping/num_total_gs": model_size,
                       "Mapping/num_new_gs": new_pts_num,
                       "Mapping/per_iteration_time": iter_opt_time,
                       "Mapping/psnr_render": opt_dict["psnr_render"],
                       "Mapping/color_loss": opt_dict[frame_id]["color_loss"],
                       "Mapping/depth_loss": opt_dict[frame_id]["depth_loss"]})

    def vis_mapping_iteration(self, frame_id, iter, color, depth, gt_color, gt_depth, seeding_mask=None, interval=10) -> None:
        """
        Visualization of depth, color images and save to file.

        Args:
            frame_id (int): current frame index.
            iter (int): the iteration number.
            save_rendered_image (bool): whether to save the rgb image in separate folder
            img_dir (str): the directory to save the visualization.
            seeding_mask: used in mapper when adding gaussians, if not none.
        """
        if not self.VERBOSE:
            return
        if frame_id % interval != 0:
            return
        gt_depth_np = gt_depth.cpu().numpy()
        gt_color_np = gt_color.cpu().numpy()

        depth_np = depth.detach().cpu().numpy()
        color = torch.round(color * 255.0) / 255.0
        color_np = color.detach().cpu().numpy()
        depth_residual = np.abs(gt_depth_np - depth_np)
        depth_residual[gt_depth_np == 0.0] = 0.0
        # make errors >=5cm noticeable
        depth_residual = np.clip(depth_residual, 0.0, 0.05)

        color_residual = np.abs(gt_color_np - color_np)
        color_residual[np.squeeze(gt_depth_np == 0.0)] = 0.0

        # Determine Aspect Ratio and Figure Size
        aspect_ratio = color.shape[1] / color.shape[0]
        fig_height = 8
        # Adjust the multiplier as needed for better spacing
        fig_width = fig_height * aspect_ratio * 1.2

        fig, axs = plt.subplots(2, 3, figsize=(fig_width, fig_height))
        axs[0, 0].imshow(gt_depth_np, cmap="jet", vmin=0, vmax=6)
        axs[0, 0].set_title('Input Depth', fontsize=16)
        axs[0, 0].set_xticks([])
        axs[0, 0].set_yticks([])
        axs[0, 1].imshow(depth_np, cmap="jet", vmin=0, vmax=6)
        axs[0, 1].set_title('Rendered Depth', fontsize=16)
        axs[0, 1].set_xticks([])
        axs[0, 1].set_yticks([])
        axs[0, 2].imshow(depth_residual, cmap="plasma")
        axs[0, 2].set_title('Depth Residual', fontsize=16)
        axs[0, 2].set_xticks([])
        axs[0, 2].set_yticks([])
        gt_color_np = np.clip(gt_color_np, 0, 1)
        color_np = np.clip(color_np, 0, 1)
        color_residual = np.clip(color_residual, 0, 1)
        axs[1, 0].imshow(gt_color_np, cmap="plasma")
        axs[1, 0].set_title('Input RGB', fontsize=16)
        axs[1, 0].set_xticks([])
        axs[1, 0].set_yticks([])
        axs[1, 1].imshow(color_np, cmap="plasma")
        axs[1, 1].set_title('Rendered RGB', fontsize=16)
        axs[1, 1].set_xticks([])
        axs[1, 1].set_yticks([])
        if seeding_mask is not None:
            axs[1, 2].imshow(seeding_mask, cmap="gray")
            axs[1, 2].set_title('Seeding Mask', fontsize=16)
            axs[1, 2].set_xticks([])
            axs[1, 2].set_yticks([])
        else:
            axs[1, 2].imshow(color_residual, cmap="plasma")
            axs[1, 2].set_title('RGB Residual', fontsize=16)
            axs[1, 2].set_xticks([])
            axs[1, 2].set_yticks([])

        for ax in axs.flatten():
            ax.axis('off')
        fig.tight_layout()
        plt.subplots_adjust(top=0.90)  # Adjust top margin
        fig_name = str(self.output_path / "mapping_vis" / f'{frame_id:04d}_{iter:04d}.jpg')
        fig_title = f"Mapper Color/Depth at frame {frame_id:04d} iters {iter:04d}"
        plt.suptitle(fig_title, y=0.98, fontsize=20)
        plt.savefig(fig_name, dpi=250, bbox_inches='tight')
        plt.clf()
        plt.close()
        if self.use_wandb:
            log_title = "Mapping_vis/" + f'{frame_id:04d}_{iter:04d}'
            wandb.log({log_title: [wandb.Image(fig_name)]})
        print(f"Saved rendering vis of color/depth at {frame_id:04d}_{iter:03d}.jpg")
