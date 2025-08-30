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
#include "../datastructures/imgpyramidrgbd.h"
#include <fstream>
#include <memory>
#include <mutex>
#include <opencv2/opencv.hpp>
#include <queue>

#ifdef WITH_REALSENSE
#include "realsensesensor.h"
#endif

class IOWrapperSettings {
public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  IOWrapperSettings(const std::string &settingsFile);
  int SKIP_FIRST_N_FRAMES;
  int READ_N_IMAGES;
  float DEPTH_SCALE_FACTOR;
  bool useDepthTimeStamp;
  // bool DO_ADAPT_CANNY_VALUES;
  bool DO_WAIT_AUTOEXP;
  cv::Size2i imgSize;
  std::string MainFolder;
  std::string subDataset;
  std::string associateFile;
  std::string poseOutDir = "./result";
  bool DO_OUTPUT_IMAGES;
  bool READ_FROM_ASTRA_DATA;
  bool EXTERNAL_INPUT;
};

struct Frame {
  cv::Mat rgb, depth;
  double timestamp;

  Frame(cv::Mat rgb, cv::Mat depth, double t)
      : rgb(rgb), depth(depth), timestamp(t){};
};

class IOWrapperRGBD {
private:
  std::ifstream fileList;
  std::ofstream associateFile;
  cv::Mat rgb, depth;
  int nFrames = 0;
  std::queue<std::unique_ptr<ImgPyramidRGBD>> mPyrQueue;
  std::mutex mtx;
  bool mFinish = false;
  bool mAllImagesRead = false;
  bool mHasMoreImages = false;

  std::string outputImgDir;

  bool readNextFrame(cv::Mat &rgb, cv::Mat &depth, double &rgbTimeStamp,
                     double &depthTimeStamp, int skipFrames,
                     double depthScaleFactor);
  int noFrames = 0;

public:
  void generateImgPyramidFromFiles(void);
  void prepareImagesFromFiles(void);
  IOWrapperRGBD(const IOWrapperSettings &settings,
                const ImgPyramidSettings &mPyrSettings,
                const std::shared_ptr<CameraPyr> &camPyr);
  inline bool isImgPyramidAvailable() {
    // std::unique_lock<std::mutex> lock(this->mtx);
    I3D_LOG(i3d::detail) << "isImgPyramidAvailable";
    return mPyrQueue.size() > 0;
  }
  inline bool hasMoreImages() { return mHasMoreImages; }
  void waitReadImages(void);
  bool getOldestPyramid(ImgPyramidRGBD &pyr);
  bool getOldestPyramid(std::shared_ptr<ImgPyramidRGBD> &pyr);
  void setFinish(bool setFinish);

  IOWrapperSettings mSettings;
  ImgPyramidSettings mPyrConfig;
  std::shared_ptr<CameraPyr> mCamPyr;
  std::queue<Frame> queue_frame;
};
