# EAGS-SLAM: Edge-Assisted Gaussian Splatting SLAM

## Abstract
Dense simultaneous localization and mapping (SLAM) based on 3D Gaussian Splatting excels at constructing detailed and realistic scenes. Some existing methods reduce cumulative drift errors and achieve globally consistent mapping by introducing loop closure. With the introduction of more components, the system's tracking accuracy and rendering quality have improved, but its speed is unsatisfactory. Therefore, we propose EAGS-SLAM, an edge-assisted RGB-D Gaussian Splatting SLAM. EAGS-SLAM combines hand-crafted edge-based visual odometry and gaussian tracking for fast coarse-to-fine camera pose estimation. The mapping backend uses edge images to assist in gaussian seeding. Our system uses a decoupled submap system that can run loop closure in parallel, ensuring globally consistent mapping while improving system speed. We evaluated our system on real-world datasets ScanNet, TUM RGB-D, and synthetic dataset Replica. Compared to some existing state-of-the-art systems, our system is competitive or superior in terms of tracking, rendering, and system speed.

## Key contributions
- We introduce EAGS-SLAM, a dense edge-assisted RGB-D Gaussian Splatting SLAM system. We combine edge-based visual odometry and gaussian tracking to reduce tracking time. We also use edge information to assist in anisotropic gaussian seeding to improve rendering quality.

- We propose a parallel loop closure method based on an improved submap system, decoupling submaps from the world coordinate system. This enables loop closure to run in parallel with tracking and mapping threads, further improving system speed.

- We evaluate our system on synthetic and real-world datasets. Experimental results show that our system is competitive or superior in tracking accuracy, rendering quality, and runtime efficiency.

## Note
The code will be soon available.