/**
 * This file is part of REVO.
 *
 * Copyright (C) 2014-2017 Schenk Fabian <schenk at icg dot tugraz dot at> (Graz
 * University of Technology) For more information see
 * <https://github.com/fabianschenk/REVO/>
 *
 * REVO is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * REVO is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with REVO. If not, see <http://www.gnu.org/licenses/>.
 */
#pragma once
#include "tracker.h"
#include <string>

#ifdef WITH_PANGOLIN_VIEWER
#include "../gui/Viewer.h"
#endif
#include "../datastructures/imgpyramidrgbd.h"
#include "../io/iowrapperRGBD.h"
#include <memory.h>

class REVOConfig {
public:
  REVOConfig(const std::string &settingsFile, const std::string &dataSettings)
      : settingsPyr(dataSettings), settingsIO(dataSettings),
        settingsTracker(settingsFile) {
    I3D_LOG(i3d::debug) << "REVOConfig reading setting file: " << settingsFile;
    cv::FileStorage dataF(settingsFile, cv::FileStorage::READ);
    if (!dataF.isOpened()) {
      I3D_LOG(i3d::error) << "REVOConfig could not open setting file: "
                          << settingsFile;
      exit(EXIT_FAILURE);
    }

    cv::read(dataF["GT_POSEREADER_ACTIVE"], GT_POSEREADER_ACTIVE, false);
    cv::read(dataF["INIT_FROM_LAST_POSE"], INIT_FROM_LAST_POSE, true);
    cv::read(dataF["DO_OUTPUT_POSES"], DO_OUTPUT_POSES, false);
    cv::read(dataF["DO_RECORD_IMAGES"], DO_RECORD_IMAGES, false);
    cv::read(dataF["DO_USE_PANGOLIN_VIEWER"], DO_USE_PANGOLIN_VIEWER, true);
    cv::read(dataF["DO_SHOW_DEBUG_IMAGE"], DO_SHOW_DEBUG_IMAGE, false);
    cv::read(dataF["DO_GENERATE_DENSE_PCL"], DO_GENERATE_DENSE_PCL, false);
    dataF.release();

    I3D_LOG(i3d::debug) << "GT_POSEREADER_ACTIVE = " << GT_POSEREADER_ACTIVE;
    I3D_LOG(i3d::debug) << "INIT_FROM_LAST_POSE = " << INIT_FROM_LAST_POSE;
    I3D_LOG(i3d::debug) << "DO_OUTPUT_POSES = " << DO_OUTPUT_POSES;
    I3D_LOG(i3d::debug) << "DO_RECORD_IMAGES = " << DO_RECORD_IMAGES;
    I3D_LOG(i3d::debug) << "DO_USE_PANGOLIN_VIEWER = "
                        << DO_USE_PANGOLIN_VIEWER;
    I3D_LOG(i3d::debug) << "DO_SHOW_DEBUG_IMAGE = " << DO_SHOW_DEBUG_IMAGE;
    I3D_LOG(i3d::debug) << "DO_GENERATE_DENSE_PCL = " << DO_GENERATE_DENSE_PCL;
  }
  // only relevant when keyframes are used
  bool INIT_FROM_LAST_POSE;
  // number of frames that should be skipped (only relevant for pre-computed
  // edges) unsigned int skipFirstFrames;
  bool GT_POSEREADER_ACTIVE;
  std::string dataFolder;
  std::string subDataset;
  std::string outputFolder;
  ImgPyramidSettings settingsPyr;
  // Eigen::Matrix3f K;
  // double distThreshold, angleThreshold;
  bool READ_FROM_ASTRA;
  bool DO_OUTPUT_POSES;
  bool DO_USE_PANGOLIN_VIEWER;
  bool DO_SHOW_DEBUG_IMAGE;
  bool DO_GENERATE_DENSE_PCL;
  IOWrapperSettings settingsIO;
  TrackerSettings settingsTracker;
  bool DO_RECORD_IMAGES;
  // WindowedOptimizationSettings settingsWO;
};

class REVO {
public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  class Pose {
  public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    Pose(const Eigen::Matrix4f T_kf_curr, double timestamp,
         const std::shared_ptr<ImgPyramidRGBD> &_kfFrame)
        : T_kf_curr(T_kf_curr), timestamp(timestamp), kfFrame(_kfFrame) {}
    inline const Eigen::Matrix4f T_N_W() const {
      return (kfFrame->getT_w_f() * T_kf_curr).inverse();
    }
    inline const Eigen::Matrix4f T_W_N() const { return getT_w_curr(); }

    inline const Eigen::Matrix4f T_N_kf() const {
      return this->T_kf_curr.inverse();
    }
    inline const Eigen::Matrix4f &T_kf_N() const { return this->T_kf_curr; }
    inline const Eigen::Matrix4f getTransKFtoCurr() const {
      return this->T_kf_curr.inverse();
    }
    inline const Eigen::Matrix4f &getTransCurrtoKF() const {
      return this->T_kf_curr;
    }
    const Eigen::Matrix4f &getPose() const { return this->T_kf_curr; }
    // world pose of current frame is:
    // T_W_curr =  T_W_KF * T_KF_CURR
    inline const Eigen::Matrix4f getT_w_curr() const {
      return kfFrame->getT_w_f() * T_kf_curr;
    }
    inline double getTimestamp() const { return timestamp; }
    inline void setT_kf_curr(const Eigen::Matrix4f trans) { T_kf_curr = trans; }
    inline void setT_w_curr(const Eigen::Matrix4f T_w_c) {
      T_kf_curr = kfFrame->getT_w_f().inverse() * T_w_c;
    }

    // this method is only called when the "previous frame" is taken as
    // keyframe!
    void setKfFrame(const std::shared_ptr<ImgPyramidRGBD> &_kfFrame) {
      // Update parent keyframe -> this is typically "itself"
      kfFrame = _kfFrame;
      // no relative transformation to "itself"
      T_kf_curr = Eigen::Matrix4f::Identity();
    }

  private:
    Eigen::Matrix4f T_kf_curr;
    double timestamp;
    std::shared_ptr<ImgPyramidRGBD> kfFrame; // this is the parent frame
  };

  std::vector<Pose, Eigen::aligned_allocator<REVO::Pose>> mPoseGraph;

  REVO(const std::string &settingsFile, const std::string &dataSettings);
  ~REVO();

  bool start(void);
  void run(void);
  Eigen::Matrix4f step(const cv::Mat &rgb, const cv::Mat &depth,
                       const double timestamp);
  inline bool isKeyFrame(void);
  void setPose(const int frame_number, const Eigen::Matrix4f);
  Eigen::Matrix4f getPose(const int frame_number);
  cv::Mat getEdgeImage(const int frame_number);
  void report(void);
  float reprojectPCLToImg(const Eigen::MatrixXf &pcl, const Eigen::Matrix3f &R,
                          const Eigen::Vector3f &T, const cv::Mat &img,
                          const cv::Size2i &size, const Eigen::Matrix3f &K,
                          int &goodCount, int &badCount,
                          const std::string &title) const;

private:
  const REVOConfig mSettings;
  std::shared_ptr<CameraPyr> camPyr;
  std::unique_ptr<TrackerNew> mTracker;
#ifdef WITH_PANGOLIN_VIEWER
  std::shared_ptr<REVOGui::MapDrawer> mpMapDrawer;
  std::shared_ptr<REVOGui::Viewer> mpViewer;
#endif
  std::unique_ptr<std::thread> mThViewer, mThIOWrapper;
  std::shared_ptr<IOWrapperRGBD> mIOWrapper;
  // reference/key frame, previous frame, current frame
  std::shared_ptr<ImgPyramidRGBD> kfPyr, prevPyr, currPyr;
  Eigen::Matrix4f T_NM1_N = Eigen::Matrix4f::Identity();
  Eigen::Matrix4f startPose = Eigen::Matrix4f::Identity();
  Eigen::Matrix3f R = Eigen::Matrix3f::Identity();
  Eigen::Vector3f T = Eigen::Vector3f(0, 0, 0);
  TrackerNew::TrackerStatus trackerStatus = TrackerNew::TRACKER_STATE_UNKNOWN;
  bool justAddedNewKeyframe = false;
  float error = INFINITY;
  int noFrames = 0;
  int nKeyFrames = 0;
  std::vector<std::tuple<Eigen::Matrix4f, bool>> baPoseGraph;
  std::vector<double> trackingTimes;
  std::vector<double> accStructureTimes;
  std::vector<double> dT;
  std::vector<cv::Mat> edge_img;
  std::vector<std::shared_ptr<ImgPyramidRGBD>> his_img_pyramid;

  inline Eigen::Matrix4f transformFromRT(const Eigen::Matrix3f &R,
                                         const Eigen::Vector3f &T) const;
  TrackerNew::TrackerStatus
  checkTrackingResult(const Eigen::Matrix4f &currPoseInWorld);
  inline std::string RTToTUMString(const Eigen::Matrix3d &R,
                                   const Eigen::Vector3d &T, double timeStamp);
  inline std::string poseToTUMString(const Eigen::Matrix4d &pose,
                                     const double timeStamp);
};
