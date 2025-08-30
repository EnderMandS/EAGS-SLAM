#!/bin/bash

set -e
set -o pipefail

echo -e "\nStart evaluation on EAGS-SLAM.\n"

#------------------------ 0 ------------------------
# TUM RGB-D
python run_slam.py configs/TUM_RGBD/rgbd_dataset_freiburg1_desk.yaml | tee log/tum/desk_0.log
python run_slam.py configs/TUM_RGBD/rgbd_dataset_freiburg1_desk2.yaml | tee log/tum/desk2_0.log
python run_slam.py configs/TUM_RGBD/rgbd_dataset_freiburg1_room.yaml | tee log/tum/room_0.log
python run_slam.py configs/TUM_RGBD/rgbd_dataset_freiburg2_xyz.yaml | tee log/tum/xyz_0.log
python run_slam.py configs/TUM_RGBD/rgbd_dataset_freiburg3_long_office_household.yaml | tee log/tum/fr3_0.log
# Replica
python run_slam.py configs/Replica/office0.yaml | tee log/replica/office0_0.log
python run_slam.py configs/Replica/office1.yaml | tee log/replica/office1_0.log
python run_slam.py configs/Replica/office2.yaml | tee log/replica/office2_0.log
python run_slam.py configs/Replica/office3.yaml | tee log/replica/office3_0.log
python run_slam.py configs/Replica/office4.yaml | tee log/replica/office4_0.log
python run_slam.py configs/Replica/room0.yaml | tee log/replica/room0_0.log
python run_slam.py configs/Replica/room1.yaml | tee log/replica/room1_0.log
python run_slam.py configs/Replica/room2.yaml | tee log/replica/room2_0.log
# ScanNet
python run_slam.py configs/ScanNet/scene0000_00.yaml | tee log/scannet/000_0.log
python run_slam.py configs/ScanNet/scene0059_00.yaml | tee log/scannet/059_0.log
python run_slam.py configs/ScanNet/scene0106_00.yaml | tee log/scannet/106_0.log
python run_slam.py configs/ScanNet/scene0169_00.yaml | tee log/scannet/169_0.log
python run_slam.py configs/ScanNet/scene0181_00.yaml | tee log/scannet/181_0.log
python run_slam.py configs/ScanNet/scene0207_00.yaml | tee log/scannet/207_0.log

# --------------------------- 1 ------------------------
# TUM RGB-D
python run_slam.py configs/TUM_RGBD/rgbd_dataset_freiburg1_desk.yaml | tee log/tum/desk_1.log
python run_slam.py configs/TUM_RGBD/rgbd_dataset_freiburg1_desk2.yaml | tee log/tum/desk2_1.log
python run_slam.py configs/TUM_RGBD/rgbd_dataset_freiburg1_room.yaml | tee log/tum/room_1.log
python run_slam.py configs/TUM_RGBD/rgbd_dataset_freiburg2_xyz.yaml | tee log/tum/xyz_1.log
python run_slam.py configs/TUM_RGBD/rgbd_dataset_freiburg3_long_office_household.yaml | tee log/tum/fr3_1.log
# Replica
python run_slam.py configs/Replica/office0.yaml | tee log/replica/office0_1.log
python run_slam.py configs/Replica/office1.yaml | tee log/replica/office1_1.log
python run_slam.py configs/Replica/office2.yaml | tee log/replica/office2_1.log
python run_slam.py configs/Replica/office3.yaml | tee log/replica/office3_1.log
python run_slam.py configs/Replica/office4.yaml | tee log/replica/office4_1.log
python run_slam.py configs/Replica/room0.yaml | tee log/replica/room0_1.log
python run_slam.py configs/Replica/room1.yaml | tee log/replica/room1_1.log
python run_slam.py configs/Replica/room2.yaml | tee log/replica/room2_1.log
# ScanNet
python run_slam.py configs/ScanNet/scene0000_00.yaml | tee log/scannet/000_1.log
python run_slam.py configs/ScanNet/scene0059_00.yaml | tee log/scannet/059_1.log
python run_slam.py configs/ScanNet/scene0106_00.yaml | tee log/scannet/106_1.log
python run_slam.py configs/ScanNet/scene0169_00.yaml | tee log/scannet/169_1.log
python run_slam.py configs/ScanNet/scene0181_00.yaml | tee log/scannet/181_1.log
python run_slam.py configs/ScanNet/scene0207_00.yaml | tee log/scannet/207_1.log

# --------------------------- 2 ------------------------
# TUM RGB-D
python run_slam.py configs/TUM_RGBD/rgbd_dataset_freiburg1_desk.yaml | tee log/tum/desk_2.log
python run_slam.py configs/TUM_RGBD/rgbd_dataset_freiburg1_desk2.yaml | tee log/tum/desk2_2.log
python run_slam.py configs/TUM_RGBD/rgbd_dataset_freiburg1_room.yaml | tee log/tum/room_2.log
python run_slam.py configs/TUM_RGBD/rgbd_dataset_freiburg2_xyz.yaml | tee log/tum/xyz_2.log
python run_slam.py configs/TUM_RGBD/rgbd_dataset_freiburg3_long_office_household.yaml | tee log/tum/fr3_2.log
# Replica
python run_slam.py configs/Replica/office0.yaml | tee log/replica/office0_2.log
python run_slam.py configs/Replica/office1.yaml | tee log/replica/office1_2.log
python run_slam.py configs/Replica/office2.yaml | tee log/replica/office2_2.log
python run_slam.py configs/Replica/office3.yaml | tee log/replica/office3_2.log
python run_slam.py configs/Replica/office4.yaml | tee log/replica/office4_2.log
python run_slam.py configs/Replica/room0.yaml | tee log/replica/room0_2.log
python run_slam.py configs/Replica/room1.yaml | tee log/replica/room1_2.log
python run_slam.py configs/Replica/room2.yaml | tee log/replica/room2_2.log
# ScanNet
python run_slam.py configs/ScanNet/scene0000_00.yaml | tee log/scannet/000_2.log
python run_slam.py configs/ScanNet/scene0059_00.yaml | tee log/scannet/059_2.log
python run_slam.py configs/ScanNet/scene0106_00.yaml | tee log/scannet/106_2.log
python run_slam.py configs/ScanNet/scene0169_00.yaml | tee log/scannet/169_2.log
python run_slam.py configs/ScanNet/scene0181_00.yaml | tee log/scannet/181_2.log
python run_slam.py configs/ScanNet/scene0207_00.yaml | tee log/scannet/207_2.log

echo -e "\nEvaluation on EAGA-SLAM complete."
