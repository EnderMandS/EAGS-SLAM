#include <Python.h>
#include <pybind11/pybind11.h>

#include <pybind11/eigen.h>
#include <pybind11/numpy.h>

#include "Logging.h"
#include "system.h"

namespace py = pybind11;

Eigen::Matrix4f voStepWrapper(REVO &, py::array_t<uint8_t> &rgb,
                              py::array_t<float> &depth,
                              const double timestamp);
pybind11::array_t<uint8_t> getEdgeImage(REVO &, const int frame_number);

PYBIND11_MODULE(VisualOdom, m) {
  m.doc() = "This is the class of REVO.";
  py::class_<REVO>(m, "VisualOdom")
      .def(py::init<const std::string &, const std::string &>(),
           "The first param is the general setting file "
           "config/revo_settings.yaml. The second param is the dataset config "
           "file config/dataset_tum1.yaml")

      .def("step", &voStepWrapper, "Output: Eigen::Matrix4f pose in world.")

      .def("report", &REVO::report, "Echo VO report")

      .def("setTwc", &REVO::setPose,
           "Force setting frame pose. This function must be called after "
           "step().")

      .def("getTwc", &REVO::getPose)

      .def("getEdgeImage", &getEdgeImage);
}

Eigen::Matrix4f voStepWrapper(REVO &vo, py::array_t<uint8_t> &rgb,
                              py::array_t<float> &depth,
                              const double timestamp) {
  py::buffer_info rgb_buf = rgb.request();
  py::buffer_info depth_buf = depth.request();
  cv::Mat image_color(rgb_buf.shape[0], rgb_buf.shape[1], CV_8UC3,
                      (uint8_t *)rgb_buf.ptr);
  cv::Mat image_depth(depth_buf.shape[0], depth_buf.shape[1], CV_32FC1,
                      (float *)depth_buf.ptr);
  I3D_LOG(i3d::detail) << "RGB Img rows: " << rgb_buf.shape[0]
                     << ", cols:" << rgb_buf.shape[1];
  I3D_LOG(i3d::detail) << "Dep Img rows: " << depth_buf.shape[0]
                     << ", cols:" << depth_buf.shape[1];
  cv::cvtColor(image_color, image_color, cv::COLOR_RGB2BGR);
  return vo.step(image_color.clone(), image_depth.clone(), timestamp);
}

py::array_t<uint8_t> getEdgeImage(REVO &vo, const int frame_number) {
  cv::Mat mat = vo.getEdgeImage(frame_number);
  if (!mat.isContinuous()) {
    I3D_LOG(i3d::error)
        << "cvMat2Numpy() Only continuous Mats are supported. Exit";
    exit(EXIT_FAILURE);
  }
  py::buffer_info buf_info(
      mat.data, sizeof(uint8_t), py::format_descriptor<uint8_t>::format(), 2,
      {mat.rows, mat.cols}, {mat.step[0], sizeof(uint8_t)});
  return py::array_t<uint8_t>(buf_info);
}
