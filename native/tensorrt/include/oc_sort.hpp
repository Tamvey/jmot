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
#include "structs.hpp"

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
    // algo params
    float det_thresh = 0.3f;
    int max_age = 30;
    int max_obs = 35;
    int min_hits = 3;
    float iou_threshold = 0.3f;

    float min_conf = 0.1f;
    int delta_t = 3;
    float inertia = 0.2f;
    float Q_xy_scaling = 0.01f;
    float Q_s_scaling = 0.0001f;

    // detector params
    std::string engine_path = "./yolo11s.engine";
    bool use_sahi;
    detection::SAHIParams sahi_params;
  };

  struct AssociateResults {
    std::vector<std::pair<int, int>> matched;
    std::vector<int> unmatched_trks;
    std::vector<int> unmatched_dets;
  };

  static Params fromYaml(const std::string &yaml_path);
  std::vector<shared_ptr<oc_sort::KalmanBoxTracker>> update(const cv::Mat &img);

  int frame_count() const;
  const Params &params() const;

  OcSort(const Params &params);
  ~OcSort();
  OcSort(const OcSort &) = delete;
  OcSort &operator=(const OcSort &) = delete;

private:
  Params params_;
  HungarianAlgorithm hung_algo_;
  detection::Detector detector_;

  int frame_count_ = -1;
  std::vector<std::shared_ptr<KalmanBoxTracker>> active_tracks_;

  std::vector<std::array<float, 5>> predict_tracks() const;

  AssociateResults associate(
      const std::vector<Eigen::Vector<float, STATE_DIM>> &dets,
      const std::vector<Eigen::Vector<float, MEAS_DIM>> &trks,
      float asso_threshold, const Eigen::MatrixXf &velocities,
      const std::vector<Eigen::Vector<float, MEAS_DIM + 1>> &k_observations,
      float inertia);
};

} // namespace oc_sort
