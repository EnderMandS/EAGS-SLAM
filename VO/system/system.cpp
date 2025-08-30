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

#include "system.h"
#include "../utils/timer.h"
#include <iomanip>
#include <numeric>
#include <sophus/se3.hpp>
#include <unistd.h>

REVO::REVO(const std::string &settingsFile, const std::string &dataSettings)
    : mSettings(settingsFile, dataSettings) {
  LOG_THRESHOLD(i3d::info);
#ifdef WITH_PANGOLIN_VIEWER
  if (mSettings.DO_USE_PANGOLIN_VIEWER) {
    // Create Drawers. These are used by the Viewer
    mpMapDrawer = std::shared_ptr<REVOGui::MapDrawer>(new REVOGui::MapDrawer());
    mpViewer =
        std::shared_ptr<REVOGui::Viewer>(new REVOGui::Viewer(mpMapDrawer));
    mThViewer = std::unique_ptr<std::thread>(
        new std::thread(&REVOGui::Viewer::Run, mpViewer));
  }
#endif
  camPyr = std::shared_ptr<CameraPyr>(new CameraPyr(mSettings.settingsPyr));
  mIOWrapper = std::shared_ptr<IOWrapperRGBD>(
      new IOWrapperRGBD(mSettings.settingsIO, mSettings.settingsPyr, camPyr));
  mTracker = std::unique_ptr<TrackerNew>(
      new TrackerNew(mSettings.settingsTracker, mSettings.settingsPyr));
  I3D_LOG(i3d::info) << "VO init success.";
}

REVO::~REVO() {
#ifdef WITH_PANGOLIN_VIEWER
  if (mSettings.DO_USE_PANGOLIN_VIEWER && mpViewer) {
    mpViewer->RequestFinish();
    while (!mpViewer->isFinished())
      usleep(5000);
    pangolin::DestroyWindow("REVO Map Viewer");
  }
#endif
  if (mThIOWrapper)
    mThIOWrapper->join();
  if (mThViewer)
    mThViewer->detach();
}

/**
 * @brief Processing an image step.
 *
 * @param rgb Reference to RGB image. The image will be clone()
 * @param depth Reference to depth image. The depth should be scaled. The image
 * will be clone()
 * @param timestamp Image time stamp
 * @return Eigen::Matrix4f Output the current frame pose in world.
 * T_world_current
 */
Eigen::Matrix4f REVO::step(const cv::Mat &rgb, const cv::Mat &depth,
                           const double timestamp) {
  auto t_start = Timer::getTime();

  // Image prepare
  I3D_LOG(i3d::debug) << "Precessing input images.";
  currPyr = std::shared_ptr<ImgPyramidRGBD>(
      new ImgPyramidRGBD(mIOWrapper->mPyrConfig, mIOWrapper->mCamPyr, rgb,
                         depth, timestamp));
  // his_img_pyramid.push_back(currPyr);
  currPyr->frameId = noFrames;
  edge_img.push_back(currPyr->returnOrigEdges(0)); // the edge image should be binary 0 or 255

  // First frame
  if (0 == noFrames++) {
    I3D_LOG(i3d::debug) << "First frame.";
    kfPyr = currPyr;
    prevPyr = currPyr;
    kfPyr->makeKeyframe();
    kfPyr->setTwf(startPose);
    kfPyr->prepareKfForStorage();
    mPoseGraph.push_back(Pose(startPose, timestamp, kfPyr));
    ++nKeyFrames;
    justAddedNewKeyframe = true;
    mTracker->addOldPclAndPose(kfPyr->return3DEdges(mTracker->histogramLevel),
                               Eigen::Matrix4f::Identity(), timestamp);
    I3D_LOG(i3d::debug) << "First frame return.";
    return startPose;
  }

  if (noFrames > 2) {
    // Init R,T with constant speed
    T_NM1_N = mPoseGraph.at(mPoseGraph.size() - 2).T_N_W() *
              mPoseGraph.back().T_W_N();
    Eigen::Matrix4f T_kf_pre = mPoseGraph.back().T_kf_N() * T_NM1_N;
    R = T_kf_pre.block<3, 3>(0, 0);
    T = T_kf_pre.block<3, 1>(0, 3);
  }

  // Track
  mTracker->trackFrames(R, T, error, kfPyr, currPyr);
  Eigen::Matrix4f T_kf_curr = transformFromRT(R, T),
                  T_w_curr = kfPyr->getT_w_f() * T_kf_curr;

  // Add key frame?
  trackerStatus = mTracker->assessTrackingQuality(T_w_curr, currPyr);
  if (trackerStatus == TrackerNew::TRACKER_STATE_NEW_KF &&
      false == justAddedNewKeyframe) {
    I3D_LOG(i3d::debug) << "Track bad, add new keyframe.";
    kfPyr = prevPyr;
    // kfPyr->setTwf(mPoseGraph.back().getT_w_curr());
    kfPyr->makeKeyframe();
    mPoseGraph.back().setKfFrame(kfPyr);
    nKeyFrames++;
    mTracker->clearUpPastLists();

    R = T_NM1_N.block<3, 3>(0, 0);
    T = T_NM1_N.block<3, 1>(0, 3);
    mTracker->trackFrames(R, T, error, kfPyr, currPyr);
    T_kf_curr = transformFromRT(R, T);
    T_w_curr = kfPyr->getT_w_f() * T_kf_curr;

    justAddedNewKeyframe = true;
  } else {
    justAddedNewKeyframe = false;
  }

  // Add good frame to pose graph
  currPyr->setTwf(T_w_curr);
  mPoseGraph.push_back(Pose(T_kf_curr, currPyr->getTimestamp(), kfPyr));
  mTracker->addOldPclAndPose(currPyr->return3DEdges(mTracker->histogramLevel),
                             T_w_curr, currPyr->getTimestamp());

  I3D_LOG(i3d::debug) << "Current pose : "
                      << poseToTUMString(T_w_curr.cast<double>(), timestamp);

  prevPyr = currPyr;

  auto dt = Timer::getTimeDiffMiS(t_start, Timer::getTime()) / 1000.0;
  trackingTimes.push_back(dt);
  I3D_LOG(i3d::debug) << "VO Track time: " << (int)dt << " ms.";

  return T_w_curr;
}

inline bool REVO::isKeyFrame(void) { return justAddedNewKeyframe; }

/**
 * @brief Force setting frame pose.
 * @param frame_number The frame number of the frame to be set.
 * @param T_w_curr Transform matrix T_world_current
 */
void REVO::setPose(const int frame_number, const Eigen::Matrix4f T_w_curr) {
  I3D_LOG(i3d::debug) << "Setting pose of frame: " << frame_number;
  if (0 == frame_number && 0 == noFrames) {
    startPose = T_w_curr;
    return;
  }
  if (frame_number > noFrames - 1) {
    I3D_LOG(i3d::error)
        << "setPose() frame number should not larger than latest frame.";
    return;
  }
  if (frame_number == noFrames - 1) {
    prevPyr->setTwf(T_w_curr);
  }
  mPoseGraph[frame_number].setT_w_curr(T_w_curr);
}

/**
 * @brief Get the camera pose of given frame number
 *
 * @param frame_number
 * @return Eigen::Matrix4f Twc
 */
Eigen::Matrix4f REVO::getPose(const int frame_number) {
  if (frame_number > noFrames - 1 || frame_number <= 0) {
    I3D_LOG(i3d::error) << "getPose() frame number error. Exit.";
    exit(EXIT_FAILURE);
  }
  return mPoseGraph[frame_number].getT_w_curr();
}

/**
 * @brief Get the binary edge image of specific framd number
 *
 * @param frame_number
 * @return cv::Mat A clone() image of edge
 */
cv::Mat REVO::getEdgeImage(const int frame_number) {
  return edge_img.at(frame_number).clone();
}

/**
 * @brief Echo some VO report information and write pose result to file. This
 * function should be called after all images are processed.
 */
void REVO::report(void) {
  I3D_LOG(i3d::info) << "-----VO Report-----";
  I3D_LOG(i3d::info) << "Frames Tracked: " << mPoseGraph.size();
  I3D_LOG(i3d::info) << "Keyframes Tracked: " << nKeyFrames;

  double sumDT = 0.0;
  for (size_t i = 0; i < ImgPyramidRGBD::dtTimes.size(); ++i)
    sumDT += ImgPyramidRGBD::dtTimes[i];
  I3D_LOG(i3d::info) << "Mean distance transform time: "
                     << sumDT / ImgPyramidRGBD::dtTimes.size();

  double sum = std::accumulate(trackingTimes.begin(), trackingTimes.end(), 0.0);
  I3D_LOG(i3d::info) << "Mean tracking time: " << sum / trackingTimes.size();

  if (mSettings.DO_OUTPUT_POSES) {
    std::string poseFileName;
    if (mSettings.settingsIO.EXTERNAL_INPUT) {
      poseFileName = mSettings.settingsIO.poseOutDir;
    } else {
      poseFileName = mSettings.settingsIO.poseOutDir + "/" +
                     mSettings.settingsIO.subDataset + ".txt";
    }
    std::ofstream out_file;
    out_file.open(poseFileName.c_str(), std::ios_base::out);
    if (!out_file.is_open()) {
      I3D_LOG(i3d::error) << "Could not open pose output file:" << poseFileName;
      exit(EXIT_FAILURE);
    }
    Eigen::Matrix4f T_pose;
    Eigen::Vector3f T;
    Eigen::Quaternionf Qf;
    for (auto pose : mPoseGraph) {
      T_pose = pose.getT_w_curr();
      T = T_pose.block<3, 1>(0, 3);
      Qf = Eigen::Quaternionf(T_pose.block<3, 3>(0, 0));
      out_file << std::fixed << pose.getTimestamp() << " "
               << std::setprecision(9) << T[0] << " " << T[1] << " " << T[2]
               << " " << Qf.x() << " " << Qf.y() << " " << Qf.z() << " "
               << Qf.w() << std::endl;
    }
    out_file.close();
    I3D_LOG(i3d::info) << "Poses have been saved to: " << poseFileName;
  }
}

bool REVO::start(void) {
  mThIOWrapper = std::unique_ptr<std::thread>(
      new std::thread(&IOWrapperRGBD::generateImgPyramidFromFiles, mIOWrapper));
  mIOWrapper->waitReadImages();

#ifdef WITH_PANGOLIN_VIEWER
  while ((mpViewer && !this->mpViewer->quitRequest() &&
          mSettings.DO_USE_PANGOLIN_VIEWER && mIOWrapper->hasMoreImages()) ||
         (mIOWrapper->hasMoreImages() && !mSettings.DO_USE_PANGOLIN_VIEWER))
#else
  while (mIOWrapper->hasMoreImages())
#endif
  {
    // wait for img pyramid
    if (!mIOWrapper->isImgPyramidAvailable()) {
      I3D_LOG(i3d::detail) << "ImgPyramid not available!";
      usleep(3000);
      continue;
    }

    // Requesting image pyramid
    I3D_LOG(i3d::debug) << "Requesting img pyramid";
    if (!mIOWrapper->getOldestPyramid(currPyr)) {
      I3D_LOG(i3d::error) << "Error getting img pyramid";
      continue;
    }
    I3D_LOG(i3d::debug) << "Got img pyramid";

    currPyr->frameId = noFrames;
    const double tumRefTimestamp = currPyr->getTimestamp();
    I3D_LOG(i3d::debug) << std::fixed << "tumRefTimestamp: " << tumRefTimestamp;
    if (noFrames == 0) // first frame -> keyframe
    {
      kfPyr = currPyr;
      prevPyr = currPyr;
      kfPyr->makeKeyframe();
      kfPyr->setTwf(Eigen::Matrix4f::Identity());
      kfPyr->prepareKfForStorage();
      mPoseGraph.push_back(
          Pose(Eigen::Matrix4f::Identity(), tumRefTimestamp, kfPyr));
      ++nKeyFrames;
#ifdef WITH_PANGOLIN_VIEWER
      if (mSettings.DO_USE_PANGOLIN_VIEWER) {
        Eigen::MatrixXf pcl;
        kfPyr->generateColoredPcl(0, pcl, mSettings.DO_GENERATE_DENSE_PCL);
        mpMapDrawer->addPclAndKfPoseToQueue(pcl, Eigen::Matrix4f::Identity());
        mpMapDrawer->addKeyframePoseF(Eigen::Matrix4f::Identity());
      }
#endif
      ++noFrames;
      justAddedNewKeyframe = true;
      mTracker->addOldPclAndPose(kfPyr->return3DEdges(mTracker->histogramLevel),
                                 Eigen::Matrix4f::Identity(),
                                 kfPyr->getTimestamp());
      continue;
    }
    I3D_LOG(i3d::debug) << std::fixed << "prevTime: " << prevPyr->getTimestamp()
                        << " currTime: " << currPyr->getTimestamp();
    ++noFrames;
    if (mSettings.DO_SHOW_DEBUG_IMAGE) {
      int goodCount, badCount;
      reprojectPCLToImg(currPyr->return3DEdges(0), R, T, kfPyr->returnEdges(0),
                        camPyr->at(0).returnSize(), kfPyr->returnK(0),
                        goodCount, badCount, "init");
    }
    auto beginTracking = Timer::getTime();
    trackerStatus = mTracker->trackFrames(R, T, error, kfPyr, currPyr);
    // positive cases
    Eigen::Matrix4f T_KF_N = transformFromRT(R, T);
    Eigen::Matrix4f currPoseInWorld = kfPyr->getT_w_f() * T_KF_N;
    // Eigen::Matrix4f currPoseInWorld =
    // relativePoseToWorld(T_KF_N);//globPoseKf.back()*pose_rc;//.inverse(); now
    // check the tracking results there are basically three options:
    //- Everything ok
    //- New keyframe needed
    //- Tracking lost
    trackerStatus = mTracker->assessTrackingQuality(
        currPoseInWorld, currPyr); //(mSettings.CHECK_TRACKING_RESULTS ?  :
                                   // TrackerNew::TRACKER_STATE_OK);
    // if tracking gets inaccurate, take the previous frame as keyframe and try
    // to optimize again. The idea is that the transformation between
    // consecutive frames is more accurate
    I3D_LOG(i3d::detail) << "posegraph: " << mPoseGraph.back().getT_w_curr();
    if (trackerStatus == TrackerNew::TRACKER_STATE_NEW_KF &&
        !justAddedNewKeyframe) {
      I3D_LOG(i3d::info) << "Track bad, add new keyframe.";
      kfPyr = prevPyr;
      kfPyr->setTwf(mPoseGraph.back().getT_w_curr());
      kfPyr->makeKeyframe();
      mPoseGraph.back().setKfFrame(kfPyr);
      nKeyFrames++;
      mTracker->clearUpPastLists();
      // now, retrack
      R = T_NM1_N.block<3, 3>(0, 0);
      T = T_NM1_N.block<3, 1>(0, 3);
      if (mSettings.DO_SHOW_DEBUG_IMAGE) {
        int goodCount, badCount;
        reprojectPCLToImg(currPyr->return3DEdges(0), R, T,
                          kfPyr->returnEdges(0), camPyr->at(0).returnSize(),
                          kfPyr->returnK(0), goodCount, badCount,
                          "init with new kf");
      }
      // track again
      mTracker->trackFrames(R, T, error, kfPyr, currPyr);
      T_KF_N = transformFromRT(R, T);
      // T_KF_NM1 = Eigen::Matrix4f::Identity();
      // relativePoseToWorld(T_KF_N);//kfPyr->getT_w_f()*T_KF_N;;
      currPoseInWorld = kfPyr->getT_w_f() * T_KF_N;
      // trackerStatus = mTracker->assessTrackingQuality(currPoseInWorld,
      // currPyr);
#ifdef WITH_PANGOLIN_VIEWER
      if (mSettings.DO_USE_PANGOLIN_VIEWER) {
        Eigen::MatrixXf pcl;
        kfPyr->generateColoredPcl(0, pcl, mSettings.DO_GENERATE_DENSE_PCL);
        mpMapDrawer->addPclAndKfPoseToQueue(
            pcl,
            kfPyr->getT_w_f()); // mKeyframes.back()->getT_w_f());
        mpMapDrawer->addKeyframePoseF(kfPyr->getT_w_f());
      }
#endif
      justAddedNewKeyframe = true;
    } else
      justAddedNewKeyframe = false;
    auto endTracking = Timer::getTime();
    trackingTimes.push_back(
        double(Timer::getTimeDiffMs(beginTracking, endTracking)));
    if (mSettings.DO_SHOW_DEBUG_IMAGE) {
      int goodCount, badCount;
      reprojectPCLToImg(currPyr->return3DEdges(0), R, T, kfPyr->returnEdges(0),
                        camPyr->at(0).returnSize(), kfPyr->returnK(0),
                        goodCount, badCount, "after tracking");
    }

    // add good frame to pose graph
    mPoseGraph.push_back(Pose(T_KF_N, currPyr->getTimestamp(), kfPyr));
    mTracker->addOldPclAndPose(currPyr->return3DEdges(mTracker->histogramLevel),
                               currPoseInWorld, currPyr->getTimestamp());
    // Description:
    // It's easy, when you think about it:
    // [KF] [1] [2] [3] [4]
    // T_KF_1_init = I
    // T_KF_2_init = T_KF_1*T_KF_1
    // T_KF_3_init = T_KF_2*T_1_2 = T_KF_2 * inv(T_KF_1) * T_KF_2
    // relative camera motion between current frame N and the frame before N-1
    T_NM1_N = mPoseGraph.at(mPoseGraph.size() - 2).T_N_W() *
              mPoseGraph.back().T_W_N();
    I3D_LOG(i3d::detail) << "T_kf_N" << T_KF_N << " vs " << T_NM1_N << " "
                         << mPoseGraph.back().T_W_N() << " "
                         << mPoseGraph.back().T_kf_N();
    const Eigen::Matrix4f T_init = mPoseGraph.back().T_kf_N() * T_NM1_N;
    R = T_init.block<3, 3>(0, 0);
    T = T_init.block<3, 1>(0, 3);
    Eigen::Matrix4f absPose =
        mPoseGraph.back().getT_w_curr(); // startPose*currPoseInWorld;
    I3D_LOG(i3d::debug) << "MY abs pose: "
                        << poseToTUMString(absPose.cast<double>(),
                                           tumRefTimestamp);
#ifdef WITH_PANGOLIN_VIEWER
    if (mSettings.DO_USE_PANGOLIN_VIEWER) {
      mpMapDrawer->addFramePoseF(currPoseInWorld);
      mpMapDrawer->SetCurrentCameraPose(currPoseInWorld.inverse());
    }
#endif
    prevPyr = currPyr;
  } // while (mIOWrapper->hasMoreImages())

  report();
  mIOWrapper->setFinish(true);
#ifdef WITH_PANGOLIN_VIEWER
  while (mpViewer && !this->mpViewer->quitRequest() &&
         mSettings.DO_USE_PANGOLIN_VIEWER) {
    usleep(3000);
  }
#endif
  return true;
}

void REVO::run(void) {
  mIOWrapper->prepareImagesFromFiles();
  int total_frame_length = mIOWrapper->queue_frame.size();

  while (!mIOWrapper->queue_frame.empty()) {
    auto frame = mIOWrapper->queue_frame.front();
    mIOWrapper->queue_frame.pop();
    int image_number = total_frame_length - mIOWrapper->queue_frame.size();
    I3D_LOG(i3d::info) << "Processing frame: " << image_number << "/"
                       << total_frame_length;

    step(frame.rgb, frame.depth, frame.timestamp);
  }
  report();
}

float REVO::reprojectPCLToImg(const Eigen::MatrixXf &pcl,
                              const Eigen::Matrix3f &R,
                              const Eigen::Vector3f &T, const cv::Mat &img,
                              const cv::Size2i &size, const Eigen::Matrix3f &K,
                              int &goodCount, int &badCount,
                              const std::string &title = "") const {
  const float fx = K(0, 0), fy = K(1, 1);
  const float cx = K(0, 2), cy = K(1, 2);
  float totalCost = 0;
  goodCount = 0;
  badCount = 0;
  cv::Mat imgTmp = img.clone();

  for (int ir = 0; ir < pcl.cols(); ir++) {
    const Eigen::VectorXf pt = pcl.col(ir).head<3>();
    Eigen::VectorXf newPt = R * pt + T;
    // std::cout << pt << std::endl;
    newPt[0] = fx * newPt[0] / newPt[2] + cx;
    newPt[1] = fy * newPt[1] / newPt[2] + cy;
    if (newPt[0] >= 0 && newPt[0] < size.width && newPt[1] >= 0 &&
        newPt[1] < size.height) {
      imgTmp.at<uint8_t>(floor(newPt[1]), floor(newPt[0])) = 100;
    }
  }
  cv::imshow(title, imgTmp);
  cv::waitKey(0);
  return totalCost;
}

inline Eigen::Matrix4f REVO::transformFromRT(const Eigen::Matrix3f &R,
                                             const Eigen::Vector3f &T) const {
  Eigen::Matrix4f transform = Eigen::Matrix4f::Identity();
  transform.block<3, 3>(0, 0) = R;
  transform.block<3, 1>(0, 3) = T;
  return transform;
}
inline std::string REVO::RTToTUMString(const Eigen::Matrix3d &R,
                                       const Eigen::Vector3d &T,
                                       double timeStamp) {
  Eigen::Quaterniond Qf(R);
  std::stringstream tumString;
  tumString << std::fixed << timeStamp << " " << T[0] << " " << T[1] << " "
            << T[2] << " " << Qf.x() << " " << Qf.y() << " " << Qf.z() << " "
            << Qf.w();
  return tumString.str();
}
inline std::string REVO::poseToTUMString(const Eigen::Matrix4d &pose,
                                         const double timeStamp) {
  Eigen::Matrix3d R = pose.block<3, 3>(0, 0);
  Eigen::Vector3d T = pose.block<3, 1>(0, 3);
  return RTToTUMString(R, T, timeStamp);
}
