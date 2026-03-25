#pragma once

#include <array>
#include <cstddef>
#include <eigen3/Eigen/Core>
#include <memory>
#include <opencv4/opencv2/opencv.hpp>
#include <string>
#include <vector>

#include "detector.hpp"
#include "kalman_box_tracker.hpp"

namespace oc_sort {

struct TrackOutput {
  std::vector<float> data;
};

class OcSort {
public:
  struct Params {
    float det_thresh = 0.3f;
    int max_age = 30;
    int max_obs = 35;
    int min_hits = 3;
    float iou_threshold = 0.5f;
    bool per_class = false;
    int nr_classes = 1;

    // OcSort-specific
    float min_conf = 0.1f;
    int delta_t = 3;
    float inertia = 0.2f;
    float Q_xy_scaling = 0.01f;
    float Q_s_scaling = 0.0001f;
    std::string engine_path = "./yolo11s.engine";
  };

  OcSort(const Params &params);

  ~OcSort();

  std::vector<TrackOutput>
  update(const std::vector<Eigen::Vector<float, 6>> &dets, const cv::Mat &img);

  void reset();

  int frame_count() const;
  const Params &params() const;

  OcSort(const OcSort &) = delete;
  OcSort &operator=(const OcSort &) = delete;

private:
  detection::Detector detector_;
  Params params_;

  int frame_count_;
  std::vector<std::shared_ptr<KalmanBoxTracker>> active_tracks_;

  void capture_init_args();

  std::vector<std::array<float, 5>> predict_tracks() const;

  static std::vector<std::pair<int, int>> associate_first(
      const cv::Mat &dets, const std::vector<std::array<float, 5>> &trks,
      const std::string &asso_func, float asso_threshold,
      const std::vector<std::array<float, 2>> &velocities,
      const std::vector<std::vector<std::array<float, 5>>> &k_observations,
      float inertia, int image_w, int image_h);

  std::shared_ptr<KalmanBoxTracker>
  create_tracker_from_det(const std::vector<float> &det, int det_ind);
};

} // namespace oc_sort
