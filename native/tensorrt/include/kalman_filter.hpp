#pragma once

#include <cstddef>
#include <deque>
#include <optional>

#include <eigen3/Eigen/Core>

namespace oc_sort {

static constexpr int STATE_DIM = 7;
static constexpr int MEAS_DIM = 4;

struct SavedFreezeState {
  Eigen::Vector<float, STATE_DIM> x;
  Eigen::Matrix<float, STATE_DIM, STATE_DIM> P;
};

class KalmanFilter {
public:
  KalmanFilter(int max_obs = 50);

  void predict();
  void update(std::optional<Eigen::Vector<float, MEAS_DIM>> obj = {});
  void freeze();
  void unfreeze();

  Eigen::Vector<float, STATE_DIM> x, x_prior, x_post;
  Eigen::Vector<float, MEAS_DIM> y;

  Eigen::Matrix<float, STATE_DIM, STATE_DIM> P, P_prior, P_post, Q, F, I;
  Eigen::Matrix<float, MEAS_DIM, STATE_DIM> H;
  Eigen::Matrix<float, MEAS_DIM, MEAS_DIM> R, S, SI;
  Eigen::Matrix<float, STATE_DIM, MEAS_DIM> M, K;

  float alpha_sq;
  int max_obs;
  bool observed;
  std::deque<std::optional<Eigen::Vector<float, MEAS_DIM>>> history_obs;
  Eigen::Vector<float, MEAS_DIM> last_measurement;
  std::optional<SavedFreezeState> saved_state;
};

} // namespace oc_sort

oc_sort::KalmanFilter::KalmanFilter(int max_obs) {
  this->max_obs = max_obs;
  alpha_sq = 1.;
  observed = false;

  x = Eigen::Vector<float, STATE_DIM>::Zero();
  x_prior = Eigen::Vector<float, STATE_DIM>::Zero();
  x_post = Eigen::Vector<float, STATE_DIM>::Zero();

  I = Eigen::Matrix<float, STATE_DIM, STATE_DIM>::Identity();

  P = Eigen::Matrix<float, STATE_DIM, STATE_DIM>::Identity();
  P_prior = Eigen::Matrix<float, STATE_DIM, STATE_DIM>::Identity();
  P_post = Eigen::Matrix<float, STATE_DIM, STATE_DIM>::Identity();

  Q = Eigen::Matrix<float, STATE_DIM, STATE_DIM>::Identity();
  F = Eigen::Matrix<float, STATE_DIM, STATE_DIM>::Identity();

  H = Eigen::Matrix<float, MEAS_DIM, STATE_DIM>::Zero();
  R = Eigen::Matrix<float, MEAS_DIM, MEAS_DIM>::Identity();
  M = Eigen::Matrix<float, STATE_DIM, MEAS_DIM>::Zero();

  K = Eigen::Matrix<float, STATE_DIM, MEAS_DIM>::Zero();
}

void oc_sort::KalmanFilter::predict() {
  x = F * x;
  P = alpha_sq * (F * P * F.transpose()) + Q;
  x_prior = x;
  P_prior = P;
}

void oc_sort::KalmanFilter::update(
    const std::optional<Eigen::Vector<float, MEAS_DIM>> obj) {
  history_obs.push_back(obj);
  if (!obj.has_value()) {
    if (observed) {
      saved_state = {x, P};
    }
    observed = false;
    x_post = x;
    P_post = P;
    y = Eigen::Vector<float, MEAS_DIM>::Zero();
    return;
  }
  if (!observed) {
    observed = true;
    unfreeze();
  }

  y = obj.value() - H * x;

  auto PHT = P * H.transpose();

  S = H * PHT + R;
  SI = S.inverse();

  K = PHT * SI;

  x = x + K * y;

  auto I_KH = I - K * H;
  P = I_KH * P * I_KH.transpose() + K * R * K.transpose();

  x_post = x;
  P_post = P;

  history_obs.push_back(obj.value());
}

void oc_sort::KalmanFilter::unfreeze() {
  if (!saved_state.has_value())
    return;
  // restore kalman state after we now detection
  x = saved_state.value().x;
  P = saved_state.value().P;

  int i1 = -1, i2 = -1;

  // get indices of last detections which are not None
  for (int i = history_obs.size() - 1; i >= 0; i--) {
    if (history_obs.at(i).has_value()) {
      if (i2 == -1)
        i2 = i;
      else if (i1 == -1)
        i1 = i;
      if (i1 != -1 && i2 != -1)
        break;
    }
  }
  assert(i1 != -1);
  assert(i2 != -1);

  auto box1 = history_obs.at(i1);
  auto box2 = history_obs.at(i2);

  auto x1 = box1.value()(0);
  auto y1 = box1.value()(1);
  auto s1 = box1.value()(2);
  auto r1 = box1.value()(3);
  auto w1 = sqrt(s1 * r1);
  auto h1 = sqrt(s1 / r1);

  auto x2 = box2.value()(0);
  auto y2 = box2.value()(1);
  auto s2 = box2.value()(2);
  auto r2 = box2.value()(3);
  auto w2 = sqrt(s2 * r2);
  auto h2 = sqrt(s2 / r2);
  auto time_gap = i2 - i1;
  auto dx = (x2 - x1) / time_gap;
  auto dy = (y2 - y1) / time_gap;

  auto dw = (w2 - w1) / time_gap;
  auto dh = (h2 - h1) / time_gap;

  // remove None and last detection
  for (int i = 0; i < time_gap; i++)
    history_obs.pop_back();
  for (int i = 0; i < time_gap; i++) {
    float w = w1 + (i + 1) * dw;
    float h = h1 + (i + 1) * dh;
    Eigen::Vector4f new_box = {x1 + (i + 1) * dx, y1 + (i + 1) * dy, w * h,
                               w / h};
    update(new_box);
    if (i != time_gap - 1)
      predict();
  }
}