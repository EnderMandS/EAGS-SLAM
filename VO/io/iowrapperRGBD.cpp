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
#include "iowrapperRGBD.h"
#include "../utils/timer.h"
#include <boost/filesystem.hpp>
#include <unistd.h>

IOWrapperSettings::IOWrapperSettings(const std::string &settingsFile) {
  I3D_LOG(i3d::debug) << "IOWrapperSettings reading setting file: "
                      << settingsFile;
  cv::FileStorage fs(settingsFile, cv::FileStorage::READ);
  if (!fs.isOpened()) {
    I3D_LOG(i3d::error) << "Couldn't open settings file at location: "
                        << settingsFile;
    exit(EXIT_FAILURE);
  }

  cv::read(fs["Camera.width"], imgSize.width, 640);
  cv::read(fs["Camera.height"], imgSize.height, 480);
  cv::read(fs["DO_WAIT_AUTOEXP"], DO_WAIT_AUTOEXP, false);
  cv::read(fs["DO_RECORD_IMAGES"], DO_OUTPUT_IMAGES, false);
  cv::read(fs["EXTERNAL_INPUT"], EXTERNAL_INPUT, false);
  poseOutDir = (std::string)fs["poseOutDir"];
  subDataset = (std::string)fs["Datasets"];

  if (!EXTERNAL_INPUT) {
    cv::read(fs["DEPTH_SCALE_FACTOR"], DEPTH_SCALE_FACTOR, 5000.0f);
    cv::read(fs["READ_N_IMAGES"], READ_N_IMAGES, 100000);
    cv::read(fs["SKIP_FIRST_N_FRAMES"], SKIP_FIRST_N_FRAMES, 0);
    cv::read(fs["useDepthTimeStamp"], useDepthTimeStamp, true);
    associateFile = (std::string)fs["ASSOCIATE"];
    MainFolder = (std::string)fs["MainFolder"];

    if (SKIP_FIRST_N_FRAMES < 20 && DO_WAIT_AUTOEXP) {
      SKIP_FIRST_N_FRAMES = 20;
      I3D_LOG(i3d::warning) << "Skip first 20 frames to avoid auto exposure!";
    }
    if (associateFile.empty()) {
      I3D_LOG(i3d::error) << "Please provide associate file.";
      exit(EXIT_FAILURE);
    }

    I3D_LOG(i3d::debug) << "DEPTH_SCALE_FACTOR = " << DEPTH_SCALE_FACTOR;
    I3D_LOG(i3d::debug) << "READ_N_IMAGES = " << READ_N_IMAGES;
    I3D_LOG(i3d::debug) << "SKIP_FIRST_N_FRAMES = " << SKIP_FIRST_N_FRAMES;
    I3D_LOG(i3d::debug) << "useDepthTimeStamp = " << useDepthTimeStamp;
    I3D_LOG(i3d::debug) << "MainFolder = " << MainFolder;
  }
  fs.release();

  I3D_LOG(i3d::debug) << "DO_WAIT_AUTOEXP = " << DO_WAIT_AUTOEXP;
  I3D_LOG(i3d::debug) << "DO_RECORD_IMAGES = " << DO_OUTPUT_IMAGES;
  I3D_LOG(i3d::debug) << "EXTERNAL_INPUT = " << EXTERNAL_INPUT;
  I3D_LOG(i3d::debug) << "poseOutDir: " << poseOutDir;
}

IOWrapperRGBD::IOWrapperRGBD(const IOWrapperSettings &settings,
                             const ImgPyramidSettings &mPyrSettings,
                             const std::shared_ptr<CameraPyr> &camPyr)
    : mSettings(settings), mPyrConfig(mPyrSettings), mCamPyr(camPyr) {
  I3D_LOG(i3d::debug) << "camPyr->size(): " << camPyr->size();

  if (!settings.EXTERNAL_INPUT) {
    fileList.open((mSettings.MainFolder + "/" + mSettings.subDataset + "/" +
                   mSettings.associateFile)
                      .c_str(),
                  std::ios_base::in);
    I3D_LOG(i3d::info) << "Reading: "
                       << (mSettings.MainFolder + "/" + mSettings.subDataset +
                           "/" + mSettings.associateFile);
    if (!fileList.is_open()) {
      I3D_LOG(i3d::error) << "Could not open associateFile at: "
                          << (mSettings.MainFolder + "/" +
                              mSettings.subDataset + "/" +
                              mSettings.associateFile);
      exit(EXIT_FAILURE);
    }
  }
  rgb = cv::Mat(mSettings.imgSize.width, mSettings.imgSize.height, CV_8UC3);
  depth = cv::Mat(mSettings.imgSize.width, mSettings.imgSize.height, CV_32FC1);
}

void IOWrapperRGBD::generateImgPyramidFromFiles(void) {
  if (mSettings.EXTERNAL_INPUT) {
    I3D_LOG(i3d::error) << "Can not generateImgPyramidFromFiles as "
                           "EXTERNAL_INPUT have been set to true.";
    exit(EXIT_FAILURE);
  }

  double rgbTimeStamp = 0, depthTimeStamp = 0;
  std::string line;
  int cnt_total_frames = 0;
  fileList.clear();
  fileList.seekg(0, std::ios::beg);
  while (std::getline(fileList, line)) {
    // Ignore lines that start with '#'
    if (line.empty() || line[0] == '#') {
      continue;
    }
    cnt_total_frames++;
  }
  fileList.clear();
  fileList.seekg(0, std::ios::beg);
  I3D_LOG(i3d::info) << "Total frames:" << cnt_total_frames;
  I3D_LOG(i3d::debug) << "mSettings.SKIP_FIRST_N_FRAMES = "
                      << mSettings.SKIP_FIRST_N_FRAMES;
  I3D_LOG(i3d::debug) << "mSettings.DEPTH_SCALE_FACTOR = "
                      << mSettings.DEPTH_SCALE_FACTOR;

  while (readNextFrame(rgb, depth, rgbTimeStamp, depthTimeStamp,
                       mSettings.SKIP_FIRST_N_FRAMES,
                       mSettings.DEPTH_SCALE_FACTOR) &&
         !mFinish) {
    const double tumRefTimestamp =
        (mSettings.useDepthTimeStamp ? depthTimeStamp : rgbTimeStamp);
    nFrames++;
    I3D_LOG(i3d::debug) << "Reading at frame:" << nFrames << "/"
                        << cnt_total_frames;
    auto t_start = Timer::getTime();
    I3D_LOG(i3d::trace) << "Before img pyramid!";
    std::unique_ptr<ImgPyramidRGBD> pyrPtr(
        new ImgPyramidRGBD(mPyrConfig, mCamPyr, rgb, depth, tumRefTimestamp));
    I3D_LOG(i3d::trace) << "Creating pyramid: "
                        << Timer::getTimeDiffMiS(t_start, Timer::getTime())
                        << " ms." << mSettings.DEPTH_SCALE_FACTOR;
    {
      auto t_Push = Timer::getTime();
      std::unique_lock<std::mutex> lock(this->mtx);
      mPyrQueue.push(std::move(pyrPtr));
      I3D_LOG(i3d::trace) << "Push wait: "
                          << Timer::getTimeDiffMiS(t_Push, Timer::getTime())
                          << " ms.";
    }
    if (nFrames > mSettings.READ_N_IMAGES)
      break;
    usleep(1000);
  }
  // this->mHasMoreImages = false;
  {
    std::unique_lock<std::mutex> lock(this->mtx);
    mAllImagesRead = true;
  }
  while (!mFinish) {
    usleep(3000);
  }
}
bool IOWrapperRGBD::readNextFrame(cv::Mat &rgb, cv::Mat &depth,
                                  double &rgbTimeStamp, double &depthTimeStamp,
                                  int skipFrames, double depthScaleFactor) {
  I3D_LOG(i3d::trace) << "Read next frame.";
  auto start = Timer::getTime();
  std::string currRGBFile, currDepthFile;
  std::string inputLine;
  // read lines
  while ((std::getline(fileList, inputLine))) {
    // ignore comments
    if (inputLine[0] == '#' || inputLine.empty())
      continue;
    noFrames++;
    if (noFrames <= skipFrames)
      continue;
    std::istringstream is_associate(inputLine);
    is_associate >> rgbTimeStamp >> currRGBFile >> depthTimeStamp >>
        currDepthFile;
    currRGBFile =
        mSettings.MainFolder + "/" + mSettings.subDataset + "/" + currRGBFile;
    currDepthFile =
        mSettings.MainFolder + "/" + mSettings.subDataset + "/" + currDepthFile;
    I3D_LOG(i3d::debug) << "RGB Files: " << currRGBFile;
    I3D_LOG(i3d::debug) << "Depth Files: " << currDepthFile;
    break;
  }

  if (fileList.eof()) {
    I3D_LOG(i3d::info) << "All frames have been read.";
    return false;
  }

  rgb = cv::imread(currRGBFile);
  depth = cv::imread(currDepthFile, CV_LOAD_IMAGE_UNCHANGED);

  bool exit_flag = false;
  if (rgb.empty()) {
    I3D_LOG(i3d::error) << "Fail to read rgb: " << currRGBFile;
    exit_flag = true;
  }
  if (depth.empty()) {
    I3D_LOG(i3d::error) << "Fail to read depth: " << currDepthFile;
    exit_flag = true;
  }
  if (exit_flag) {
    exit(EXIT_FAILURE);
  }

  depth.convertTo(depth, CV_32FC1, 1.0f / depthScaleFactor);
  // divide by 5000 to get distance in metres
  // depth = depth/depthScaleFactor;
  auto dt = std::chrono::duration_cast<std::chrono::microseconds>(
                Timer::getTime() - start)
                .count();
  I3D_LOG(i3d::trace) << "Read time: " << dt << "ms";
  return true;
}

void IOWrapperRGBD::prepareImagesFromFiles(void) {
  if (mSettings.EXTERNAL_INPUT) {
    I3D_LOG(i3d::error) << "Can not prepareImagesFromFiles as "
                           "EXTERNAL_INPUT have been set to true.";
    exit(EXIT_FAILURE);
  }

  I3D_LOG(i3d::info) << "Preparing images from file.";
  auto t_start = Timer::getTime();
  double rgbTimeStamp = 0, depthTimeStamp = 0;
  std::string line;

  I3D_LOG(i3d::debug) << "mSettings.SKIP_FIRST_N_FRAMES = "
                      << mSettings.SKIP_FIRST_N_FRAMES;
  I3D_LOG(i3d::debug) << "mSettings.DEPTH_SCALE_FACTOR = "
                      << mSettings.DEPTH_SCALE_FACTOR;
  std::string rgb_file_name, depth_file_name;
  std::string inputLine;
  while ((std::getline(fileList, inputLine))) {
    if (inputLine[0] == '#' || inputLine.empty())
      continue;
    noFrames++;
    if (noFrames <= mSettings.SKIP_FIRST_N_FRAMES)
      continue;

    std::istringstream is_associate(inputLine);
    is_associate >> rgbTimeStamp >> rgb_file_name >> depthTimeStamp >>
        depth_file_name;
    rgb_file_name =
        mSettings.MainFolder + "/" + mSettings.subDataset + "/" + rgb_file_name;
    depth_file_name = mSettings.MainFolder + "/" + mSettings.subDataset + "/" +
                      depth_file_name;

    I3D_LOG(i3d::debug) << "Reading RGB image: " << rgb_file_name;
    I3D_LOG(i3d::debug) << "Reading depth image: " << depth_file_name;
    auto rgb = cv::imread(rgb_file_name);
    auto depth = cv::imread(depth_file_name, CV_LOAD_IMAGE_UNCHANGED);
    depth.convertTo(depth, CV_32FC1, 1.0f / mSettings.DEPTH_SCALE_FACTOR);

    if (rgb.empty() || depth.empty()) {
      I3D_LOG(i3d::error) << "Fail to read image." << rgb_file_name;
      exit(EXIT_FAILURE);
    }

    queue_frame.push(
        Frame(rgb, depth,
              mSettings.useDepthTimeStamp ? depthTimeStamp : rgbTimeStamp));

    if (fileList.eof()) {
      I3D_LOG(i3d::debug) << "All list read.";
      break;
    }
  }

  auto dt = Timer::getTimeDiffMs(t_start, Timer::getTime());
  I3D_LOG(i3d::debug) << "Load time: " << dt << "ms";
  I3D_LOG(i3d::info) << "Loaded " << queue_frame.size() << "/" << noFrames
                     << " images.";
}

void IOWrapperRGBD::setFinish(bool setFinish) {
  std::unique_lock<std::mutex> lock(this->mtx);
  this->mFinish = true;
}

void IOWrapperRGBD::waitReadImages(void) {
  I3D_LOG(i3d::info) << "Loading images...";
  bool flag = false;
  while (!flag) {
    sleep(1);
    {
      std::unique_lock<std::mutex> lock(this->mtx);
      flag = mAllImagesRead;
    }
  }
}

bool IOWrapperRGBD::getOldestPyramid(std::shared_ptr<ImgPyramidRGBD> &pyr) {

  I3D_LOG(i3d::trace) << "getOldestPyramid = " << mPyrQueue.size();
  if (mPyrQueue.empty())
    return false;
  I3D_LOG(i3d::trace) << "mPyrQueue.size() = " << mPyrQueue.size();
  std::unique_lock<std::mutex> lock(this->mtx);
  pyr = std::move(mPyrQueue.front());
  mPyrQueue.pop();
  if (mPyrQueue.empty() && mAllImagesRead)
    this->mHasMoreImages = false;
  return pyr != NULL;
}
