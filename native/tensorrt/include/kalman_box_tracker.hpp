#pragma once

#include <cmath>
#include <deque>
#include <optional>

#include <eigen3/Eigen/Core>
#include <eigen3/Eigen/Dense>

#include "kalman_filter.hpp"

namespace oc_sort {

inline Eigen::Vector2f speed_direction(Eigen::Vector4f bbox1,
                                       Eigen::Vector4f bbox2) {
  auto cx1 = (bbox1[0] + bbox1[2]) / 2.0;
  auto cy1 = (bbox1[1] + bbox1[3]) / 2.0;
  auto cx2 = (bbox2[0] + bbox2[2]) / 2.0;
  auto cy2 = (bbox2[1] + bbox2[3]) / 2.0;
  auto norm =
      sqrt((cy2 - cy1) * (cy2 - cy1) + (cx2 - cx1) * (cx2 - cx1)) + 1e-6;
  Eigen::Vector2f speed = {(cy2 - cy1) / norm, (cx2 - cx1) / norm};
  return speed;
}

inline Eigen::Vector<float, 4>
convert_x_to_bbox(const Eigen::Vector<float, 7> &x) {
  auto w = sqrt(x[2] * x[3]);
  auto h = x[2] / w;
  return {x[0] - w / 2.0, x[1] - h / 2.0, x[0] + w / 2.0, x[1] + h / 2.0};
}

inline Eigen::Vector<float, MEAS_DIM> xywh2xysr(const Eigen::Vector4f &bbox) {
  float x1 = bbox(0), y1 = bbox(1), w = bbox(2), h = bbox(3);
  float x = x1 + w / 2.0f;
  float y = y1 + h / 2.0f;
  float s = w * h;
  float r = w / h;

  Eigen::Vector<float, MEAS_DIM> obs;
  obs << x, y, s, r;
  return obs;
}

inline Eigen::Vector<float, MEAS_DIM> xyxy2xysr(const Eigen::Vector4f &bbox) {
  float x1 = bbox(0), y1 = bbox(1), x2 = bbox(2), y2 = bbox(3);
  float x = (x1 + x2) / 2.0f;
  float y = (y1 + y2) / 2.0f;
  float w = x2 - x1;
  float h = y2 - y1;
  float s = w * h;
  float r = w / h;

  Eigen::Vector<float, MEAS_DIM> obs;
  obs << x, y, s, r;
  return obs;
}

class KalmanBoxTracker {
public:
  static int count;

  KalmanBoxTracker(const Eigen::Vector<float, 7> &bbox, int cls, int det_ind,
                   float delta_t = 3.0f, int max_obs = 50,
                   float Q_xy_scaling = 0.01f, float Q_s_scaling = 0.0001f);

  Eigen::Vector<float, MEAS_DIM> predict();
  void update(std::optional<Eigen::Vector4f> bbox, float confidence, int cls,
              int det_ind);

  // Геттеры
  int get_id() const { return id; }
  int get_cls() const { return cls; }
  float get_conf() const { return conf; }
  int get_time_since_update() const { return time_since_update; }
  int get_hits() const { return hits; }
  int get_hit_streak() const { return hit_streak; }
  int get_age() const { return age; }
  Eigen::Vector<float, STATE_DIM> get_state() const { return kf.x; }

  Eigen::Vector4f get_last_observation() const { return last_observation; }
  std::optional<Eigen::Vector2f> get_velocity() const { return velocity; }

  KalmanFilter kf;

private:
  int id;
  int det_ind;
  int cls;
  float conf;

  int time_since_update;
  int hits;
  int hit_streak;
  int age;

  float delta_t;
  int max_obs;

  Eigen::Vector<float, MEAS_DIM> last_observation;
  std::deque<Eigen::Vector<float, MEAS_DIM>> history_observations;
  std::unordered_map<int, Eigen::Vector4f> observations;
  std::optional<Eigen::Vector2f> velocity;

  float Q_xy_scaling;
  float Q_s_scaling;
};

int KalmanBoxTracker::count = 0;

KalmanBoxTracker::KalmanBoxTracker(const Eigen::Vector<float, 7> &bbox, int cls,
                                   int det_ind, float delta_t, int max_obs,
                                   float Q_xy_scaling, float Q_s_scaling)
    : det_ind(det_ind), cls(cls), conf(bbox(5)), delta_t(delta_t),
      max_obs(max_obs), Q_xy_scaling(Q_xy_scaling), Q_s_scaling(Q_s_scaling),
      time_since_update(0), hits(0), hit_streak(0), age(0), kf(max_obs) {
  kf.F(0, 4) = 1.0f;
  kf.F(1, 5) = 1.0f;
  kf.F(2, 6) = 1.0f;

  kf.H = Eigen::Matrix<float, MEAS_DIM, STATE_DIM>::Zero();
  for (int i = 0; i < MEAS_DIM; ++i) {
    kf.H(i, i) = 1.0f;
  }

  kf.R.block<2, 2>(2, 2) *= 10.0f;

  kf.P.block<3, 3>(4, 4) *= 1000.0f;

  kf.P *= 10.0f;

  kf.Q.block<2, 2>(4, 4) *= Q_xy_scaling;

  kf.Q(6, 6) *= Q_s_scaling;

  kf.x.head<MEAS_DIM>() = xywh2xysr(bbox.head<MEAS_DIM>());

  last_observation = Eigen::Vector<float, MEAS_DIM>::Constant(-1.0f);

  id = count++;
}

Eigen::Vector<float, MEAS_DIM> KalmanBoxTracker::predict() {
  if (kf.x[6] + kf.x[2] <= 0)
    kf.x[6] *= 0;
  kf.predict();
  age += 1;
  if (time_since_update > 0)
    hit_streak = 0;
  time_since_update++;
  history_observations.push_back(convert_x_to_bbox(kf.x));
  return history_observations[history_observations.size() - 1];
}

void KalmanBoxTracker::update(
    std::optional<Eigen::Vector<float, MEAS_DIM>> bbox, float confidence,
    int cls, int det_ind) {
  this->det_ind = det_ind;
  if (bbox.has_value()) {
    conf = confidence;
    this->cls = cls;
    if (last_observation.sum() >= 0) {
      std::optional<Eigen::Vector4f> previous_box;
      for (int i = 0; i < delta_t; i++) {
        auto dt = delta_t - i;
        if (observations.find(age - dt) != observations.end()) {
          previous_box = observations[age - dt];
        }
      }
      if (!previous_box.has_value())
        previous_box = last_observation;
      velocity = speed_direction(previous_box.value(), bbox.value());
    }
    last_observation = bbox.value();
    observations[age] = bbox.value();
    history_observations.push_back(bbox.value());
    time_since_update = 0;
    hits++;
    hit_streak++;
    kf.update(xywh2xysr(bbox.value()));

  } else {
    kf.update({});
  }
}

} // namespace oc_sort