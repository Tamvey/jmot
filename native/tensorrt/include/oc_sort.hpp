#pragma once

#include <algorithm>
#include <array>
#include <cstddef>
#include <eigen3/Eigen/Core>
#include <memory>
#include <opencv4/opencv2/opencv.hpp>
#include <string>
#include <vector>

#include "Eigen/src/Core/Matrix.h"
#include "Hungarian.h"
#include "detector.hpp"
#include "kalman_box_tracker.hpp"
#include "kalman_filter.hpp"

namespace oc_sort {

struct DirectionResults {
  Eigen::MatrixXf dy;
  Eigen::MatrixXf dx;
};

Eigen::MatrixXf iou_batch(const Eigen::MatrixXf &dets,
                          const Eigen::MatrixXf &trks);

DirectionResults speed_direction_batch(const Eigen::MatrixXf &dets,
                                       const Eigen::MatrixXf &trks);

Eigen::Vector<float, MEAS_DIM + 1> k_previous_obs(
    std::unordered_map<int, Eigen::Vector<float, MEAS_DIM + 1>> observations,
    int age, int delta_t);

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
    std::string engine_path = "./models/yolo11s_16.engine";
  };

  struct AssociateResults {
    std::vector<std::pair<int, int>> matched;
    std::vector<int> unmatched_trks;
    std::vector<int> unmatched_dets;
  };

  OcSort(const Params &params);

  ~OcSort();

  std::vector<shared_ptr<oc_sort::KalmanBoxTracker>> update(const cv::Mat &img);

  void reset();

  int frame_count() const;
  const Params &params() const;

  OcSort(const OcSort &) = delete;
  OcSort &operator=(const OcSort &) = delete;

private:
  detection::Detector detector_;
  Params params_;
  HungarianAlgorithm hung_algo;

  int frame_count_;
  std::vector<std::shared_ptr<KalmanBoxTracker>> active_tracks_;

  void capture_init_args();

  std::vector<std::array<float, 5>> predict_tracks() const;

  AssociateResults associate(
      const std::vector<Eigen::Vector<float, 7>> &dets,
      const std::vector<Eigen::Vector<float, 4>> &trks, float asso_threshold,
      const Eigen::MatrixXf &velocities,
      const std::vector<Eigen::Vector<float, MEAS_DIM + 1>> &k_observations,
      float inertia);

  std::shared_ptr<KalmanBoxTracker>
  create_tracker_from_det(const std::vector<float> &det, int det_ind);
};

} // namespace oc_sort
