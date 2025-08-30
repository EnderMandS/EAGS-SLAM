from argparse import ArgumentParser
import torch
import numpy as np
import roma
from pathlib import Path

from src.entities.gaussian_model import GaussianModel
from src.entities.arguments import OptimizationParams
from src.utils.io_utils import save_dict_to_ckpt

class Submap():
    def __init__(self):
        self.device:int = None
        self.verbose:bool = None
        self.id:int = None
        self.gaussian_params:dict = None
        self.kf_id:torch.Tensor = None
        self.Tmc:torch.Tensor = None
        self.T_prev_m:torch.Tensor = None

    def load(self, path:str|Path, device:str|int="cpu", verbose:bool=False):
        """ 
        Load submap from file
        Args:
            path (str|Path): Path to file
            device (str|int): Where the submap to load
            verbose (bool): verbose
        """
        submap_dict = torch.load(path, map_location=torch.device(device))
        self.device = device
        self.verbose = verbose
        self.gaussian_params:dict = submap_dict['gaussian_params']
        self.id = submap_dict["id"]
        self.kf_id = submap_dict["kf_id"]
        self.Tmc = submap_dict["Tmc"]
        self.T_prev_m = submap_dict["T_prev_m"]
        return self
    
    def restore_gauss(self, T_prev:torch.Tensor, update_param_only:bool=False) -> GaussianModel:
        """ 
        Restore gaussian model from the loaded submap.

        Args:
            T_prev (torch.Tensor): The previous frame pose under world coordinate T_world_previous
            update_param_only (bool): Return None is set to True, else return the resotred GaussianModel

        Returns:
            GaussianModel
        """
        if self.gaussian_params is None:
            raise Exception("Load submap before restoring gaussian model")
        
        Twm = T_prev.type(torch.float64) @ self.T_prev_m
        xyz = self.gaussian_params["xyz"].type(torch.float64, non_blocking=True)
        q = torch.nn.functional.normalize(self.gaussian_params["rotation"]).type(torch.float64, non_blocking=True)
        torch.cuda.synchronize(torch.device(self.device))

        T_gauss = torch.eye(4, dtype=torch.float64, device=self.device).repeat(xyz.shape[0], 1, 1)
        T_gauss[:, :3, 3] = xyz
        T_gauss[:, :3, :3] = roma.unitquat_to_rotmat(q)
        T_gauss = Twm @ T_gauss
        self.gaussian_params["xyz"] = T_gauss[:, :3,3].type(torch.float, non_blocking=True)
        self.gaussian_params["rotation"] = roma.rotmat_to_unitquat(T_gauss[:, :3, :3]).type(torch.float, non_blocking=True)
        torch.cuda.synchronize(torch.device(self.device))

        if not update_param_only:
            gaussian_model = GaussianModel(sh_degree=0, device=self.device, verbose=self.verbose)
            gaussian_model.restore_from_params(self.gaussian_params,
                OptimizationParams(ArgumentParser(description="Training script parameters")))
        return gaussian_model if not update_param_only else None

    def from_model(self, id:int, gaussian_model:GaussianModel, Twc:torch.Tensor, T_prev_m:torch.Tensor,
                   keyframes_info:dict):
        """ 
        Create a submap from gaussian model.
        Args:
            id (int): The submap id.
            gaussian_model (GaussianModel): The GaussianModel, it will be save under world coordinate.
            Twc (torch.Tensor): The poses under world coordinate. The poses will be save under submap coordinate and connect to previous submap by T_prev_m.
            T_prev_m (torch.Tensor): The transform matrix connects the submap to the previous pose.
            keyframes_info (dict): Only the key frame id will be save with sorted.
        """
        self.device = gaussian_model.device
        gaussian_params = gaussian_model.capture_dict(non_blocking=True, device=self.device)
        Twc = Twc.to(dtype=torch.float64, device=self.device, non_blocking=True)
        self.T_prev_m = T_prev_m.type(dtype=torch.float64, non_blocking=True)
        self.device = gaussian_model.device
        self.verbose = gaussian_model.VERBOSE
        self.id = id
        self.kf_id = torch.from_numpy(np.array(sorted(list(keyframes_info.keys())))).int()
        torch.cuda.synchronize(torch.device(self.device))

        xyz = gaussian_params["xyz"].type(torch.float64, non_blocking=True)
        q = torch.nn.functional.normalize(gaussian_params["rotation"]).type(torch.float64, non_blocking=True)
        self.Tmc = Twc[0].inverse() @ Twc
        torch.cuda.synchronize(torch.device(self.device))

        T_gauss = torch.eye(4, dtype=torch.float64, device=self.device).repeat(xyz.shape[0], 1, 1)
        T_gauss[:, :3, 3] = xyz
        T_gauss[:, :3, :3] = roma.unitquat_to_rotmat(q)
        T_gauss = Twc[0].inverse() @ T_gauss
        gaussian_params["xyz"] = T_gauss[:, :3, 3].type(torch.float, non_blocking=True)
        gaussian_params["rotation"] = roma.rotmat_to_unitquat(T_gauss[:, :3, :3]).type(torch.float, non_blocking=True)
        torch.cuda.synchronize(torch.device(self.device))

        self.gaussian_params = gaussian_params
        return self

    def save(self, path:str|Path) -> None:
        """
        Saving the submap

        Args:
            path (str|Path): Path to save
        """
        submap_dict = {
            "id": self.id,
            "gaussian_params": self.gaussian_params,
            "Tmc": self.Tmc,
            "kf_id": self.kf_id,
            "T_prev_m": self.T_prev_m,
        }
        save_dict_to_ckpt(
            submap_dict, f"{str(self.id).zfill(6)}.ckpt", directory=path)
