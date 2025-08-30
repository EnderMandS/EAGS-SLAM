import math
import os
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import json
import imageio
import trimesh
from tqdm import tqdm
import concurrent.futures
import threading

class BaseDataset(torch.utils.data.Dataset):
    def __init__(self, dataset_config: dict):
        super().__init__()
        self.dataset_path = Path(dataset_config["input_path"])
        self.frame_limit = dataset_config.get("frame_limit", -1)
        self.dataset_config = dataset_config
        self.height = dataset_config["H"]
        self.width = dataset_config["W"]
        self.fx = dataset_config["fx"]
        self.fy = dataset_config["fy"]
        self.cx = dataset_config["cx"]
        self.cy = dataset_config["cy"]
        self.intrinsics_origin = np.array(
            [[self.fx, 0, self.cx], [0, self.fy, self.cy], [0, 0, 1]])

        self.depth_scale = dataset_config["depth_scale"]
        self.distortion = np.array(
            dataset_config['distortion']) if 'distortion' in dataset_config else None
        self.crop_edge:int = dataset_config['crop_edge'] if 'crop_edge' in dataset_config else 0
        if self.crop_edge:
            self.height -= 2 * self.crop_edge
            self.width -= 2 * self.crop_edge
            self.cx -= self.crop_edge
            self.cy -= self.crop_edge

        self.fovx = 2 * math.atan(self.width / (2 * self.fx))
        self.fovy = 2 * math.atan(self.height / (2 * self.fy))
        self.intrinsics = np.array(
            [[self.fx, 0, self.cx], [0, self.fy, self.cy], [0, 0, 1]])

        self.color_paths = []
        self.depth_paths = []
        self.color_images = []
        self.depth_images = []
        self.timestamps:list[float] = []
        self.poses = []

        self.executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
        self.future:concurrent.futures.Future = None
        self.cancel_event = threading.Event()

        self.loaded_index = int(0)
        self.load_lock = threading.Lock()

    def __len__(self):
        return len(self.color_paths) if self.frame_limit < 0 else int(self.frame_limit)
    
    def preload(self):
        raise NotImplementedError
    
    def get_origin_image(self, index:int):
        while not self.future.done():
            with self.load_lock:
                loaded_index = self.loaded_index
            if index < loaded_index:
                break
            time.sleep(1.0)

        color = np.array(self.color_images[index])
        depth = np.array(self.depth_images[index])
        return color, depth

    def wait_loading(self):
        if self.future:
            self.future.result()

    def __getitem__(self, index:int):
        raise NotImplementedError

class Replica(BaseDataset):
    def __init__(self, dataset_config: dict):
        super().__init__(dataset_config)
        self.color_paths = sorted(
            list((self.dataset_path / "results").glob("frame*.jpg")))
        self.depth_paths = sorted(
            list((self.dataset_path / "results").glob("depth*.png")))
        self.load_poses(self.dataset_path / "traj.txt")
        print(f"Total {len(self.color_paths) if self.frame_limit < 0 else int(self.frame_limit):d} frames.")
        self.future = self.executor.submit(self.preload)

    def load_poses(self, path):
        with open(path, "r") as f:
            lines = f.readlines()
        for line in lines:
            c2w = np.array(list(map(float, line.split()))).reshape(4, 4)
            self.poses.append(c2w.astype(np.float32))

    def preload(self):
        for i in range(len(self.color_paths) if self.frame_limit < 0 else int(self.frame_limit)):
            if self.cancel_event.is_set():
                return
            color = cv2.imread(str(self.color_paths[i]))
            depth = cv2.imread(str(self.depth_paths[i]), cv2.IMREAD_UNCHANGED)

            if color is None or depth is None:
                print("Fail to read image: " + self.color_paths[i] + " " + self.depth_paths[i])
                exit()
            
            color = cv2.cvtColor(color, cv2.COLOR_BGR2RGB)
            depth = depth.astype(np.float32) / self.depth_scale

            self.color_images.append(color)
            self.depth_images.append(depth)
            self.timestamps.append(0.1*i)

            with self.load_lock:
                self.loaded_index += 1

    def __getitem__(self, index):
        while not self.future.done():
            with self.load_lock:
                loaded_index = self.loaded_index
            if index < loaded_index:
                break
            time.sleep(1.0)

        color_data = np.array(self.color_images[index])
        depth_data = np.array(self.depth_images[index])
        return index, color_data, depth_data, self.poses[index]

class TUM_RGBD(BaseDataset):
    def __init__(self, dataset_config: dict):
        super().__init__(dataset_config)
        self.color_paths, self.depth_paths, self.poses, self.timestamps = self.loadtum(
            self.dataset_path, frame_rate=32)
        print(f"Total {len(self.color_paths) if self.frame_limit < 0 else int(self.frame_limit):d} frames.")
        self.future = self.executor.submit(self.preload)

    def parse_list(self, filepath, skiprows=0):
        """ read list data """
        return np.loadtxt(filepath, delimiter=' ', dtype=np.unicode_, skiprows=skiprows)

    def associate_frames(self, tstamp_image, tstamp_depth, tstamp_pose, max_dt=0.08):
        """ pair images, depths, and poses """
        associations = []
        for i, t in enumerate(tstamp_image):
            if tstamp_pose is None:
                j = np.argmin(np.abs(tstamp_depth - t))
                if (np.abs(tstamp_depth[j] - t) < max_dt):
                    associations.append((i, j))
            else:
                j = np.argmin(np.abs(tstamp_depth - t))
                k = np.argmin(np.abs(tstamp_pose - t))
                if (np.abs(tstamp_depth[j] - t) < max_dt) and (np.abs(tstamp_pose[k] - t) < max_dt):
                    associations.append((i, j, k))
        return associations

    def loadtum(self, datapath, frame_rate=-1):
        """ read video data in tum-rgbd format """
        if os.path.isfile(os.path.join(datapath, 'groundtruth.txt')):
            pose_list = os.path.join(datapath, 'groundtruth.txt')
        elif os.path.isfile(os.path.join(datapath, 'pose.txt')):
            pose_list = os.path.join(datapath, 'pose.txt')

        image_list = os.path.join(datapath, 'rgb.txt')
        depth_list = os.path.join(datapath, 'depth.txt')

        image_data = self.parse_list(image_list)
        depth_data = self.parse_list(depth_list)
        pose_data = self.parse_list(pose_list, skiprows=1)
        pose_vecs = pose_data[:, 1:].astype(np.float64)

        tstamp_image = image_data[:, 0].astype(np.float64)
        tstamp_depth = depth_data[:, 0].astype(np.float64)
        tstamp_pose = pose_data[:, 0].astype(np.float64)
        associations = self.associate_frames(
            tstamp_image, tstamp_depth, tstamp_pose)

        indicies = [0]
        for i in range(1, len(associations)):
            t0 = tstamp_image[associations[indicies[-1]][0]]
            t1 = tstamp_image[associations[i][0]]
            if t1 - t0 > 1.0 / frame_rate:
                indicies += [i]

        images, poses, depths, timestamp = [], [], [], []
        init_c2w = None # Make poses relative to the first pose
        for ix in indicies:
            (i, j, k) = associations[ix]
            images += [os.path.join(datapath, image_data[i, 1])]
            depths += [os.path.join(datapath, depth_data[j, 1])]
            timestamp.append(tstamp_image[i])
            w2c = self.pose_matrix_from_quaternion(pose_vecs[k])
            if init_c2w is None:
                init_c2w = np.linalg.inv(w2c)
                w2c = np.eye(4)
            else:
                w2c = init_c2w@w2c
            poses += [w2c.astype(np.float32)]

        return images, depths, poses, timestamp

    def pose_matrix_from_quaternion(self, pvec):
        """ convert 4x4 pose matrix to (t, q) """
        from scipy.spatial.transform import Rotation

        pose = np.eye(4)
        pose[:3, :3] = Rotation.from_quat(pvec[3:]).as_matrix()
        pose[:3, 3] = pvec[:3]
        return pose

    def preload(self):
        for i in range(len(self.color_paths) if self.frame_limit < 0 else int(self.frame_limit)):
            if self.cancel_event.is_set():
                return

            color = cv2.imread(self.color_paths[i]) # BGR
            depth = cv2.imread(self.depth_paths[i], cv2.IMREAD_UNCHANGED)

            if color is None or depth is None:
                tqdm.write("Fail to read image: " + self.color_paths[i] + " " + self.depth_paths[i])
                exit()

            if self.distortion is not None:
                color = cv2.undistort(color, self.intrinsics_origin, self.distortion)

            color = cv2.cvtColor(color, cv2.COLOR_BGR2RGB)
            depth = depth.astype(np.float32) / self.depth_scale

            self.color_images.append(color.copy())
            self.depth_images.append(depth.copy())

            with self.load_lock:
                self.loaded_index += 1

    def __getitem__(self, index):
        while not self.future.done():
            with self.load_lock:
                loaded_index = self.loaded_index
            if index < loaded_index:
                break
            time.sleep(1.0)

        color = self.color_images[index].copy()
        depth = self.depth_images[index].copy()
        edge = self.crop_edge
        if edge > 0:
            color = color[edge:-edge, edge:-edge].copy()
            depth = depth[edge:-edge, edge:-edge].copy()
        return index, np.array(color), np.array(depth), self.poses[index]

class ScanNet(BaseDataset):
    def __init__(self, dataset_config: dict):
        super().__init__(dataset_config)
        self.color_paths = sorted(list(
            (self.dataset_path / "rgb").glob("*.png")), key=lambda x: int(os.path.basename(x)[-9:-4]))
        self.depth_paths = sorted(list(
            (self.dataset_path / "depth").glob("*.TIFF")), key=lambda x: int(os.path.basename(x)[-10:-5]))
        self.n_img = len(self.color_paths)
        self.load_poses(self.dataset_path / "gt_pose.txt")
        print(f"Total {len(self.color_paths) if self.frame_limit < 0 else int(self.frame_limit):d} frames.")
        self.future = self.executor.submit(self.preload)

    def load_poses(self, path):
        pose_data = np.loadtxt(path, delimiter=" ", dtype=np.unicode_, skiprows=1)
        pose_vecs = pose_data[:, 0:].astype(np.float64)
        for i in range(self.n_img):
            quat = pose_vecs[i][4:]
            trans = pose_vecs[i][1:4]
            t = trimesh.transformations.quaternion_matrix(np.roll(quat, 1))
            t[:3, 3] = trans
            pose = t
            self.poses.append(pose)
            self.timestamps.append(pose_vecs[i][0])

    def preload(self):
        for i in range(len(self.color_paths) if self.frame_limit < 0 else int(self.frame_limit)):
            if self.cancel_event.is_set():
                return
            color = cv2.imread(str(self.color_paths[i]))
            depth = cv2.imread(str(self.depth_paths[i]), cv2.IMREAD_UNCHANGED)

            if color is None or depth is None:
                print("Fail to read image: " + self.color_paths[i] + " " + self.depth_paths[i])
                exit()

            if self.distortion is not None:
                color = cv2.undistort(color, self.intrinsics_origin, self.distortion)

            color = cv2.cvtColor(color, cv2.COLOR_BGR2RGB)
            depth = depth.astype(np.float32) / self.depth_scale

            self.color_images.append(color.copy())
            self.depth_images.append(depth.copy())

            with self.load_lock:
                self.loaded_index += 1

    def __getitem__(self, index):
        while not self.future.done():
            with self.load_lock:
                loaded_index = self.loaded_index
            if index < loaded_index:
                break
            time.sleep(1.0)

        color = self.color_images[index].copy()
        depth = self.depth_images[index].copy()
        edge = self.crop_edge
        if edge > 0:
            color = color[edge:-edge, edge:-edge].copy()
            depth = depth[edge:-edge, edge:-edge].copy()
        return index, np.array(color), np.array(depth), self.poses[index]

class ScanNetPP(BaseDataset):
    def __init__(self, dataset_config: dict):
        super().__init__(dataset_config)
        self.use_train_split = dataset_config["use_train_split"]
        self.train_test_split = json.load(open(f"{self.dataset_path}/dslr/train_test_lists.json", "r"))
        if self.use_train_split:
            self.image_names = self.train_test_split["train"]
        else:
            self.image_names = self.train_test_split["test"]
        self.load_data()
        print(f"Total {len(self.color_paths) if self.frame_limit < 0 else int(self.frame_limit):d} frames.")
        self.future = self.executor.submit(self.preload)

    def load_data(self):
        self.poses = []
        cams_path = self.dataset_path / "dslr" / "nerfstudio" / "transforms_undistorted.json"
        cams_metadata = json.load(open(str(cams_path), "r"))
        frames_key = "frames" if self.use_train_split else "test_frames"
        frames_metadata = cams_metadata[frames_key]
        frame2idx = {frame["file_path"]: index for index, frame in enumerate(frames_metadata)}
        P = np.array([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]]).astype(np.float32)
        for image_name in self.image_names:
            frame_metadata = frames_metadata[frame2idx[image_name]]
            # if self.ignore_bad and frame_metadata['is_bad']:
            #     continue
            color_path = str(self.dataset_path / "dslr" / "undistorted_images" / image_name)
            depth_path = str(self.dataset_path / "dslr" / "undistorted_depths" / image_name.replace('.JPG', '.png'))
            self.color_paths.append(color_path)
            self.depth_paths.append(depth_path)
            c2w = np.array(frame_metadata["transform_matrix"]).astype(np.float32)
            c2w = P @ c2w @ P.T
            self.poses.append(c2w)

    def preload(self):
        for i in range(len(self.color_paths) if self.frame_limit < 0 else int(self.frame_limit)):
            if self.cancel_event.is_set():
                return
            color_data = np.asarray(imageio.imread(self.color_paths[i]), dtype=float)
            color_data = cv2.resize(color_data, (self.width, self.height), interpolation=cv2.INTER_LINEAR)

            depth_data = np.asarray(imageio.imread(self.depth_paths[i]), dtype=np.int64)
            depth_data = cv2.resize(depth_data.astype(float), (self.width, self.height), interpolation=cv2.INTER_NEAREST)
            depth_data = depth_data.astype(np.float32) / self.depth_scale

            self.color_images.append(color_data.copy())
            self.depth_images.append(depth_data.copy())
            self.timestamps.append(0.1*i)

            with self.load_lock:
                self.loaded_index += 1

    def __len__(self):
        if self.use_train_split:
            return len(self.image_names) if self.frame_limit < 0 else int(self.frame_limit)
        else:
            return len(self.image_names)

    def __getitem__(self, index:int):
        while not self.future.done():
            with self.load_lock:
                loaded_index = self.loaded_index
            if index < loaded_index:
                break
            time.sleep(1.0)

        color_data = self.color_images[index].astype(np.uint8).copy()
        depth_data = self.depth_images[index].copy()
        return index, color_data, depth_data, self.poses[index]
    
    def get_origin_image(self, index:int):
        while not self.future.done():
            with self.load_lock:
                loaded_index = self.loaded_index
            if index < loaded_index:
                break
            time.sleep(1.0)

        color = cv2.resize(self.color_images[index].astype(np.uint8), (640, 480), interpolation=cv2.INTER_LINEAR)
        depth = cv2.resize(self.depth_images[index], (640, 480), interpolation=cv2.INTER_NEAREST)
        return color, depth

def get_dataset(dataset_name: str):
    if dataset_name == "replica":
        return Replica
    elif dataset_name == "tum":
        return TUM_RGBD
    elif dataset_name == "scannet":
        return ScanNet
    elif dataset_name == "scannetpp":
        # There is a BUG of memory leak when sending images to VO at a specific resolution.
        # The BUG is not only occurred in ScanNet++ but also in other datasets.
        # My skills are limited and can not locate the code that causing the problem.
        # I avoided this BUG by resizing the image. Though this makes the code work,
        # it seems to cause some other problems.
        # If you can find the problem, you are more than welcome to submit a PR to fix it!
        return ScanNetPP
    raise NotImplementedError(f"Dataset {dataset_name} not implemented")
