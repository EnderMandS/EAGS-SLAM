# EAGS-SLAM: Edge-Assisted Gaussian Splatting SLAM

<div align="center">
  <h3>
    <a href="https://endermands.github.io/EAGSSLAM/">Project Page</a> | 
    <a href="">Paper (Under submission)</a>
  </h3>
</div>

## Abstract
Existing 3D Gaussian Splatting (3DGS)-based Simultaneous Localization and Mapping (SLAM) methods improve tracking accuracy and rendering quality by introducing more components such as loop closure. However, this comes at the cost of reduced speed and these methods do not effectively utilize the information in the image. In this paper, we propose EAGS-SLAM, an edge-assisted RGB-D Gaussian Splatting SLAM that utilizes edge information in images to improve system performance. EAGS-SLAM combines edge-based visual odometry and gaussian tracking, reducing camera tracking time by up to half. The mapping backend uses edge information to assist in gaussian seeding to capture geometric details in the image. Our system uses a decoupled submap system that can run loop closure in parallel, ensuring globally consistent mapping while improving system speed. We evaluated our system on real-world datasets ScanNet, TUM RGB-D, and synthetic dataset Replica. Compared to some existing state-of-the-art systems, our system is competitive or superior in terms of tracking, rendering, and system speed.

## Key Features
- **Coarse-to-Fine Tracking**: Combining edge-based visual odometry and gaussian tracking to reduce tracking time
- **Edge-Assisted Gaussian Seeding**: Using edge information for gaussian seeding to improve rendering quality
- **Parallel Loop Closure**: Optimized the submap system to run loop closure in parallel, improving system speed

## Note
The code will be available after the paper has been accepted.