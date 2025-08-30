""" This module includes the Gaussian-SLAM class, which is responsible for controlling Mapper and Tracker
    It also decides when to start a new submap and when to update the estimated camera poses.
"""
import os, time
import pprint
from argparse import ArgumentParser
from datetime import datetime
from pathlib import Path
import shutil

import torch
import cv2

from src.entities.arguments import OptimizationParams
from src.entities.datasets import get_dataset
from src.entities.gaussian_model import GaussianModel
from src.entities.mapper import Mapper
from src.entities.tracker import Tracker
from src.entities.lc import Loop_closure
from src.entities.logger import Logger
from src.entities.submap import Submap
from src.utils.io_utils import save_dict_to_ckpt, save_dict_to_yaml
from src.utils.mapper_utils import exceeds_motion_thresholds 
from src.utils.utils import np2torch, setup_seed, torch2np
from src.utils.vis_utils import *  # noqa - needed for debugging

import matplotlib.pyplot as plt
from tqdm import tqdm

class GaussianSLAM(object):

    def __init__(self, config: dict) -> None:

        self._setup_output_path(config)
        self.device = config["device"]
        torch.cuda.set_device(self.device)
        self.config = config
        self.VERBOSE:bool = self.config["verbose"]

        if self.VERBOSE:
            print('Tracking config')
            pprint.PrettyPrinter().pprint(config["tracking"])
            print('Mapping config')
            pprint.PrettyPrinter().pprint(config["mapping"])
            print('Loop closure config')
            pprint.PrettyPrinter().pprint(config["lc"])

        self.scene_name = config["data"]["scene_name"]
        self.dataset_name = config["dataset_name"]
        self.dataset = get_dataset(config["dataset_name"])({**config["data"], **config["cam"]})

        n_frames = len(self.dataset)
        frame_ids = list(range(n_frames))
        self.mapping_frame_ids = frame_ids[::config["mapping"]["map_every"]] + [n_frames - 1]

        self.estimated_c2ws = torch.empty(len(self.dataset), 4, 4, dtype=torch.float)
        self.exposures_ab = torch.zeros(len(self.dataset), 2)

        save_dict_to_yaml(config, "config.yaml", directory=self.output_path)

        self.submap_using_motion_heuristic = config["mapping"]["submap_using_motion_heuristic"]

        self.keyframes_info = {}
        self.opt = OptimizationParams(ArgumentParser(description="Training script parameters"))

        self.new_submap_frame_ids = [0]

        self.logger = Logger(self.output_path, config["use_wandb"], verbose=self.VERBOSE)
        self.mapper = Mapper(config["mapping"], self.dataset, self.logger, device=self.device, verbose=self.VERBOSE)
        self.tracker = Tracker(config, self.dataset, self.logger, device=self.device)
        self.enable_exposure = self.tracker.enable_exposure
        self.LC_PARALLEL:bool = self.config["lc"]["parallel"] if "parallel" in self.config["lc"] else True
        self.loop_closer = Loop_closure(config, self.dataset, self.logger)
        self.loop_closer.submap_path = self.output_path / "submaps"

    def cleanup(self):
        if self.dataset.future is not None and not self.dataset.future.done():
            self.dataset.cancel_event.set()
            self.dataset.future.result()
        self.loop_closer.executor.shutdown(wait=True, cancel_futures=True)

    def _setup_output_path(self, config: dict) -> None:
        """ Sets up the output path for saving results based on the provided configuration. If the output path is not
        specified in the configuration, it creates a new directory with a timestamp.
        Args:
            config: A dictionary containing the experiment configuration including data and output path information.
        """
        if "output_path" not in config["data"]:
            output_path = Path(config["data"]["output_path"])
            self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.output_path = output_path / self.timestamp
        else:
            self.output_path = Path(config["data"]["output_path"])

        if os.path.exists(self.output_path):
            shutil.rmtree(self.output_path)
        self.output_path.mkdir(parents=True)
        os.makedirs(self.output_path / "mapping_vis")
        os.makedirs(self.output_path / "tracking_vis")
        os.makedirs(self.output_path / "poses")
        os.makedirs(self.output_path / "submaps")

    def should_start_new_submap(self, frame_id: int) -> bool:
        """ Determines whether a new submap should be started based on the motion heuristic or specific frame IDs.
        Args:
            frame_id: The ID of the current frame being processed.
        Returns:
            A boolean indicating whether to start a new submap.
        """
        if self.submap_using_motion_heuristic:
            if exceeds_motion_thresholds(
                self.estimated_c2ws[frame_id], self.estimated_c2ws[self.new_submap_frame_ids[-1]],
                    rot_thre=50, trans_thre=0.5):
                return True
        elif frame_id % self.config["mapping"]["new_submap_every"] == 0 and frame_id != 0:
            return True
        return False
     
    def save_current_submap(self, id:int, gaussian_model:GaussianModel,
                               Twc:torch.Tensor, T_prev_m:torch.Tensor):
        """
        Saving the current submap
        Args:
            id (int): The submap id
            gaussian_model (GaussianModel): The current GaussianModel instance to capture and reset for the new submap.
            Twc (torch.Tensor): The estimate poses T_map_cam of the submap
            T_prev_m (torch.Tensor): The transform matrix connects the submap to the previous pose.
        """
        Submap().from_model(id, gaussian_model, Twc, T_prev_m, self.keyframes_info) \
            .save(self.loop_closer.submap_path)
    
    def start_new_submap(self, frame_id: int) -> GaussianModel:
        """ Initializes a new submap.
        This function updates the submap count and optionally marks the current frame ID for new submap initiation.
        Args:
            frame_id: The ID of the current frame at which the new submap is started.
        Returns:
            GaussianModel: A new, reset GaussianModel instance for the new submap.
        """
        gaussian_model = GaussianModel(0, device=self.device, verbose=self.config["verbose"])
        gaussian_model.training_setup(self.opt)
        self.mapper.keyframes = []
        self.keyframes_info = {}
        self.new_submap_frame_ids.append(frame_id)
        self.mapping_frame_ids.append(frame_id) if frame_id not in self.mapping_frame_ids else self.mapping_frame_ids
        self.submap_id += 1
        return gaussian_model
    
    def report(self):
        '''
        Plot the tracker iteration figures and save.
        '''
        if self.config["verbose"]:
            plt.clf()
            plt.hist(self.tracker.iter_cnt, bins=10)
            plt.xlabel("iteration")
            plt.ylabel("frame")
            plt.title("Track Iteration Count")
            plt.savefig(os.path.join(self.output_path, "tracking_vis", "iter.png"))
            plt.clf()

            plt.hist(self.tracker.iter_cnt_min_loss, bins=10)
            plt.xlabel("iteration to min loss")
            plt.ylabel("frame")
            plt.title("Track Iteration Count to Min Loss")
            plt.savefig(os.path.join(self.output_path, "tracking_vis", "iter_min_loss.png"))
            plt.clf()

        print(f"\nTotal {len(self.new_submap_frame_ids)} submaps.")
        print(f"New submap at frame id:{self.new_submap_frame_ids}\n")

        self.tracker.report()
        self.mapper.report()

    def run(self) -> None:
        """ Starts the main program flow for Gaussian-SLAM, including tracking and mapping. """
        torch.cuda.empty_cache()
        setup_seed(self.config["seed"])
        gaussian_model = GaussianModel(0, device=self.device, verbose=self.config["verbose"])
        gaussian_model.training_setup(self.opt)
        self.submap_id = int(0)

        exposure_ab = torch.nn.Parameter(torch.tensor(
            0.0, device=self.device)), torch.nn.Parameter(torch.tensor(0.0, device=self.device))
        
        track_time = []
        map_time = []
        len_dataset = len(self.dataset)

        # Set VO initial pose
        if self.tracker.help_camera_initialization or self.tracker.odometry_type == "odometer":
            self.tracker.vo.setTwc(0, self.dataset.poses[0])
        
        total_t_start = time.perf_counter()
        for frame_id in range(len_dataset) if self.config["verbose"] else \
            tqdm(range(len_dataset), desc="Processing frames", unit="frame"):
            # print(f"Start processing frame {frame_id}", flush=True)
            if frame_id in [0, 1]:
                estimated_c2w = self.dataset.poses[frame_id]
                exposure_ab = torch.nn.Parameter(torch.tensor(
                    0.0, device=self.device)), torch.nn.Parameter(torch.tensor(0.0, device=self.device))
                if self.tracker.help_camera_initialization or self.tracker.odometry_type == "odometer":
                    image, depth = self.dataset.get_origin_image(frame_id)
                    self.tracker.vo.step(image, depth, self.dataset.timestamps[frame_id])
                    if frame_id!=0:
                        self.tracker.vo.setTwc(frame_id, estimated_c2w)
            else:
                t_start = time.perf_counter()
                estimated_c2w, exposure_ab = self.tracker.track(frame_id, gaussian_model,
                    torch2np(self.estimated_c2ws[torch.tensor([frame_id-2, frame_id-1])]))
                dt = (time.perf_counter()-t_start) * 1000 # ms
                track_time.append(dt)
                if self.config["verbose"]:
                    print(f"Track time: {dt:.2f} ms.", flush=True)
            
            exposure_ab = exposure_ab if self.enable_exposure else None
            self.estimated_c2ws[frame_id] = np2torch(estimated_c2w)

            # Reinitialize gaussian model for new segment
            if frame_id<len_dataset-1 and self.should_start_new_submap(frame_id):
                if self.config["verbose"]:
                    print(f"\nNew submap at {frame_id}")
                i = self.new_submap_frame_ids[-1]
                if i==0:
                    T_prev_m = self.estimated_c2ws[0].type(torch.float64)
                else:
                    T_prev_m = self.estimated_c2ws[i-1].type(torch.float64).inverse() @ \
                                 self.estimated_c2ws[i].type(torch.float64)
                self.save_current_submap(self.submap_id, gaussian_model,
                                         self.estimated_c2ws[i:frame_id], T_prev_m)
                gaussian_model = None
                torch.cuda.empty_cache()
                future = self.loop_closer.submit(self.submap_id, frame_id)
                if not self.LC_PARALLEL:
                    future.result()
                gaussian_model = self.start_new_submap(frame_id)

            if frame_id in self.mapping_frame_ids:
                if self.config["verbose"]:
                    print(f"\nMapping frame {frame_id}", flush=True)

                gaussian_model.training_setup(self.opt, exposure_ab) 
                estimate_c2w = torch2np(self.estimated_c2ws[frame_id])
                new_submap = not bool(self.keyframes_info)
                edge_img = None
                if (self.tracker.help_camera_initialization or self.tracker.odometry_type == "odometer") \
                    and self.dataset_name != "scannetpp":
                    edge_img = self.tracker.vo.getEdgeImage(frame_id)
                    crop_edge = self.dataset.crop_edge
                    if crop_edge > 0:
                        edge_img = edge_img[crop_edge:-crop_edge, crop_edge:-crop_edge].copy()
                else:
                    edge_img = cv2.Canny(cv2.cvtColor(self.dataset[frame_id][1], cv2.COLOR_RGB2GRAY), 150, 100)

                t_start = time.perf_counter()
                opt_dict = self.mapper.map(frame_id, estimate_c2w, gaussian_model,
                                           new_submap, exposure_ab, edge_img)
                dt = (time.perf_counter()-t_start) * 1000 # ms
                map_time.append(dt)
                if self.config["verbose"]:
                    print(f"Map time: {dt:.2f} ms.\n", flush=True)

                # Keyframes info update
                self.keyframes_info[frame_id] = {
                    "keyframe_id": frame_id, 
                    "opt_dict": opt_dict,
                }
                if self.enable_exposure:
                    self.keyframes_info[frame_id]["exposure_a"] = exposure_ab[0].item()
                    self.keyframes_info[frame_id]["exposure_b"] = exposure_ab[1].item()
            
            if self.enable_exposure:
                self.exposures_ab[frame_id] = torch.tensor([exposure_ab[0].item(), exposure_ab[1].item()])
            
            self.loop_closer.check_futures()
            torch.cuda.empty_cache()

        # Final loop closure
        i = self.new_submap_frame_ids[-1]
        T_prev_m = self.estimated_c2ws[i-1].type(torch.float64).inverse() @ self.estimated_c2ws[i].type(torch.float64)
        self.save_current_submap(self.submap_id, gaussian_model,
                                    self.estimated_c2ws[i:], T_prev_m)
        gaussian_model = None
        torch.cuda.empty_cache()
        if self.config["verbose"]:
            t_start = time.perf_counter()
        print("Waiting for LC...", flush=True)
        self.loop_closer.executor.shutdown()
        if self.config["verbose"]:
            print(f"\nFinal LC wait time: {time.perf_counter()-t_start:.2f}s.")
        self.loop_closer.update_submaps_info_from_file(self.submap_id)
        if self.config['lc']['final']:
            lc_output = self.loop_closer.loop_closure(self.submap_id)
            if len(lc_output) > 0:
                self.loop_closer.apply_correction_to_submaps(lc_output)
            torch.cuda.empty_cache()

        total_time = int(time.perf_counter()-total_t_start)
        print(f"\nTotal time: {total_time}s.")
        print(f"FPS: {len_dataset/total_time:.4f}, frame process time:{total_time/len_dataset:.4f}")

        if self.enable_exposure:
            save_dict_to_ckpt(self.exposures_ab, "exposures_ab.ckpt", directory=self.output_path)
        self.loop_closer.save_Twc()

        # Report
        print(f"\nTrack time avg:{np.average(track_time):.2f}ms, " +
            f"min:{min(track_time):.2f}ms, max:{max(track_time):.2f}ms.")
        print(f"Map time avg:{np.average(map_time):.2f}ms, " +
            f"min:{min(map_time):.2f}ms, max:{max(map_time):.2f}ms.\n")
        self.report()
        if (self.tracker.help_camera_initialization or self.tracker.odometry_type == "odometer"):
            self.tracker.vo.report()

        torch.cuda.empty_cache()
