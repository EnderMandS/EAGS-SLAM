import numpy as np
import os
import cv2
from tqdm import tqdm
from scipy.spatial.transform import Rotation as R
import shutil

fps_fake = 20
raw_folder = "/mnt/sdb/dataset/scannet/data/scans"

scenes = ['scene0000_00', 'scene0059_00', 'scene0106_00', 'scene0169_00', 'scene0181_00', 'scene0207_00']

for scene_idx, scene in enumerate(scenes):
    save_folder = os.path.join(raw_folder, scene, "processed")
    data_folder = os.path.join(raw_folder, scene)
    
    os.makedirs(save_folder)
    os.makedirs(os.path.join(save_folder,"rgb"))
    os.makedirs(os.path.join(save_folder,"depth"))
    
    shutil.copy(os.path.join(data_folder, 'data/intrinsic/intrinsic_depth.txt'), 
                os.path.join(save_folder, 'intrinsic.txt'))
    
    with open(os.path.join(save_folder,'gt_pose.txt'), 'w') as f:
        f.write('# timestamp tx ty tz qx qy qz qw\n')
    
    initial_time_stamp = 0.0 
    
    color_folder = os.path.join(data_folder,"data/color")
    depth_folder = os.path.join(data_folder,"data/depth")
    pose_folder = os.path.join(data_folder,"data/pose")
    
    num_frames = len(os.listdir(color_folder))
    
    frame_idx = 0
    for raw_idx in tqdm(range(num_frames), desc="Processing frames", unit="frame"):
        with open(os.path.join(pose_folder,"{}.txt".format(raw_idx)), "r") as f:
            lines = f.readlines()
            M_w_c = np.zeros((4,4))
            for i in range(4):
                content = lines[i].split(" ")
                for j in range(4):
                    M_w_c[i,j] = float(content[j])
                    
        if "inf" in lines[0]:
            # invalid gt poses, skip this frame
            continue

        ######## convert depth to [m] and float type #########
        depth = cv2.imread(os.path.join(depth_folder,"{}.png".format(raw_idx)),cv2.IMREAD_UNCHANGED)
        depth = depth.astype("float32")/1000.0

        ######## resize rgb to the same size of depth #########
        rgb = cv2.imread(os.path.join(color_folder,"{}.jpg".format(raw_idx)))
        rgb = cv2.resize(rgb,(depth.shape[1],depth.shape[0]),interpolation=cv2.INTER_CUBIC)

        cv2.imwrite(os.path.join(save_folder,"rgb/frame_{}.png".format(str(frame_idx).zfill(5))),rgb)
        cv2.imwrite(os.path.join(save_folder,"depth/frame_{}.TIFF".format(str(frame_idx).zfill(5))),depth)

        content = "{:.4f}".format(initial_time_stamp + frame_idx*1.0/fps_fake)
        for t in M_w_c[:3,3]:
            content += " {:.9f}".format(t)
        for q in R.from_matrix(M_w_c[:3,:3]).as_quat():
            content += " {:.9f}".format(q)
        
        with open(os.path.join(save_folder,'gt_pose.txt'), 'a') as f:
            f.write(content + '\n')
            
        frame_idx += 1