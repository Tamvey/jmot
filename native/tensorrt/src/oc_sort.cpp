#include <algorithm>
#include <memory>
#include <numeric>
#include <vector>

#include <Eigen/Core>

#include "Hungarian.h"
#include "kalman_box_tracker.hpp"
#include "kalman_filter.hpp"
#include "oc_sort.hpp"
#include "structs.hpp"

std::vector<std::vector<double>> eigen_to_vector2d(const Eigen::MatrixXf &mat) {
  std::vector<std::vector<double>> result(mat.rows(),
                                          std::vector<double>(mat.cols()));

  for (int i = 0; i < mat.rows(); ++i) {
    for (int j = 0; j < mat.cols(); ++j) {
      result[i][j] = mat(i, j);
    }
  }

  return result;
}

Eigen::Vector<float, oc_sort::MEAS_DIM + 1> oc_sort::k_previous_obs(
    std::unordered_map<int, Eigen::Vector<float, MEAS_DIM + 1>> observations,
    int age, int delta_t) {
  if (observations.empty())
    return Eigen::Vector<float, MEAS_DIM + 1>::Zero();
  for (int i = 0; i < delta_t; i++) {
    auto dt = delta_t - i;
    if (observations.find(age - dt) != observations.end())
      return observations[age - dt];
  }
  auto max_age = 0;
  std::for_each(
      observations.begin(), observations.end(),
      [&max_age](std::pair<int, Eigen::Vector<float, MEAS_DIM + 1>> pair) {
        max_age = std::max(max_age, pair.first);
      });
  return observations[max_age];
}

Eigen::MatrixXf oc_sort::iou_batch(const Eigen::MatrixXf &dets,
                                   const Eigen::MatrixXf &trks) {
  int N = dets.rows();
  int M = trks.rows();
  Eigen::MatrixXf iou(N, M);

  for (int i = 0; i < N; ++i) {
    float x1d = dets(i, 0);
    float y1d = dets(i, 1);
    float x2d = dets(i, 0) + dets(i, 2);
    float y2d = dets(i, 1) + dets(i, 3);
    float aread = (x2d - x1d) * (y2d - y1d);

    for (int j = 0; j < M; ++j) {
      float x1t = trks(j, 0);
      float y1t = trks(j, 1);
      float x2t = trks(j, 0) + trks(j, 2);
      float y2t = trks(j, 1) + trks(j, 3);

      float ix1 = std::max(x1d, x1t);
      float iy1 = std::max(y1d, y1t);
      float ix2 = std::min(x2d, x2t);
      float iy2 = std::min(y2d, y2t);

      float inter = std::max(0.0f, ix2 - ix1) * std::max(0.0f, iy2 - iy1);
      float areat = (x2t - x1t) * (y2t - y1t);
      float uni = aread + areat - inter;

      iou(i, j) = inter / (uni + 1e-6f);
    }
  }
  return iou;
}

oc_sort::DirectionResults
oc_sort::speed_direction_batch(const Eigen::MatrixXf &dets,
                               const Eigen::MatrixXf &trks) {
  Eigen::VectorXf CX1 = dets.col(0) + dets.col(2) / 2;
  Eigen::VectorXf CY1 = dets.col(1) + dets.col(3) / 2;

  Eigen::VectorXf CX2 = trks.col(0) + trks.col(2) / 2;
  Eigen::VectorXf CY2 = trks.col(1) + trks.col(3) / 2;

  Eigen::MatrixXf dx = CX2 * Eigen::RowVectorXf::Ones(CX1.rows()) -
                       Eigen::VectorXf::Ones(CX2.rows()) * CX1.transpose();
  Eigen::MatrixXf dy = CY2 * Eigen::RowVectorXf::Ones(CY1.rows()) -
                       Eigen::VectorXf::Ones(CY2.rows()) * CY1.transpose();

  Eigen::MatrixXf norm = (dx.array().square() + dy.array().square()).sqrt();
  norm = norm.array() + 1e-6f;

  dx = dx.array() / norm.array();
  dy = dy.array() / norm.array();
  return {std::move(dx), std::move(dy)};
}

oc_sort::OcSort::AssociateResults oc_sort::OcSort::associate(
    const std::vector<Eigen::Vector<float, 7>> &dets,
    const std::vector<Eigen::Vector<float, 4>> &trks, float asso_threshold,
    const Eigen::MatrixXf &velocities,
    const std::vector<Eigen::Vector<float, MEAS_DIM + 1>> &k_observations,
    float inertia) {
  AssociateResults as_res{};
  DirectionResults dir_res{};

  if (trks.empty()) {
    as_res.unmatched_dets.resize(dets.size());
    std::iota(as_res.unmatched_dets.begin(), as_res.unmatched_dets.end(), 0);
    return as_res;
  }

  Eigen::MatrixXf m_dets(dets.size(), dets[0].size());
  for (int i = 0; i < m_dets.rows(); i++)
    m_dets.row(i) = dets[i].transpose();

  Eigen::MatrixXf m_obs(k_observations.size(), k_observations[0].size());
  for (int i = 0; i < m_obs.rows(); i++)
    m_obs.row(i) = k_observations[i].transpose();

  Eigen::MatrixXf m_trks(trks.size(), trks[0].size());
  for (int i = 0; i < m_trks.rows(); i++)
    m_trks.row(i) = trks[i].transpose();

  dir_res = speed_direction_batch(m_dets, m_obs);

  Eigen::VectorXf inertia_Y = velocities.col(0);
  Eigen::VectorXf inertia_X = velocities.col(1);

  Eigen::MatrixXf m_inertia_Y = inertia_Y.replicate(1, dir_res.dy.cols());

  Eigen::MatrixXf m_inertia_X = inertia_X.replicate(1, dir_res.dx.cols());

  Eigen::MatrixXf diff_angle_cos = m_inertia_X.array() * dir_res.dx.array() +
                                   m_inertia_Y.array() * dir_res.dy.array();

  diff_angle_cos = diff_angle_cos.cwiseMax(-1.0f).cwiseMin(1.0f);

  Eigen::MatrixXf diff_angle = diff_angle_cos.array().acos();
  diff_angle = (M_PI / 2.0f - diff_angle.array().abs()) / M_PI;

  // Mask
  Eigen::VectorXf valid_mask = (m_obs.col(4).array() >= 0.0f).cast<float>();
  Eigen::MatrixXf valid_mask_mat =
      valid_mask.replicate(1, dets.size()); // M x N
  // std::cout << "valid_mask_mat:\n" << valid_mask_mat << std::endl;

  // IoU matrix
  // std::cout << "m_dets:\n" << m_dets << std::endl;
  // std::cout << "m_trks:\n" << m_trks << std::endl;
  Eigen::MatrixXf iou_matrix =
      iou_batch(m_dets.leftCols(4), m_trks.leftCols(4));
  // std::cout << "iou_matrix:\n" << iou_matrix << std::endl;

  // Det scores of tracks
  Eigen::MatrixXf scores = m_dets.col(4).replicate(1, trks.size());
  // std::cout << "scores:\n" << scores << std::endl;

  // Speed cost-matrix
  Eigen::MatrixXf angle_diff_cost =
      (valid_mask_mat.array() * diff_angle.array() * inertia).matrix();
  angle_diff_cost.transposeInPlace();
  angle_diff_cost = angle_diff_cost.array() * scores.array();
  // std::cout << "angle_diff_cost:\n" << angle_diff_cost << std::endl;

  // Final cost
  Eigen::MatrixXf final_cost =
      (2.5f - iou_matrix.array() + angle_diff_cost.array()).matrix();

  // std::cout << "final_cost:\n" << final_cost << std::endl;

  // Linear assignment (Hungarian);
  std::vector<int> assignment;
  std::vector<std::vector<double>> final_cost_v = eigen_to_vector2d(final_cost);
  hung_algo.Solve(final_cost_v, assignment);

  int N = dets.size();
  int M = k_observations.size();

  std::vector<bool> det_matched(N, false);
  std::vector<bool> trk_matched(M, false);
  std::vector<std::pair<int, int>> matches;

  for (int i = 0; i < assignment.size(); i++) {
    int t = assignment[i];

    if (t >= 0 && t < M && iou_matrix(i, t) >= asso_threshold) {
      matches.emplace_back(i, t);
      det_matched[i] = true;
      trk_matched[t] = true;
    } else {
      as_res.unmatched_dets.push_back(i);
      if (t != -1)
        as_res.unmatched_trks.push_back(t);
      // det_matched[i] = true;
      // trk_matched[t] = true;
    }
  }

  as_res.matched = std::move(matches);

  for (int i = 0; i < N; i++) {
    if (!det_matched[i])
      as_res.unmatched_dets.push_back(i);
  }

  for (int i = 0; i < M; i++) {
    if (!trk_matched[i])
      as_res.unmatched_trks.push_back(i);
  }

  return as_res;
}

std::vector<shared_ptr<oc_sort::KalmanBoxTracker>>
oc_sort::OcSort::update(const cv::Mat &img) {
  auto detections = detector_.detect(img, true, 0.2, 0.2,
                                     detection::SAHIParams{320, 320, 0.2, 0.2});
  frame_count_++;

  // sieve by threshold and form detection array
  std::vector<Eigen::Vector<float, 7>> dets;
  int c = 0;
  for (const auto &det : detections) {
    // for detection vector
    Eigen::Vector<float, 7> tmp1;
    tmp1 << det.box.x, det.box.y, det.box.width, det.box.height, det.conf,
        det.classId, c++;
    dets.push_back(tmp1);
  }

  // prediction on each active track
  std::vector<Eigen::Vector<float, 4>> v_trcks;
  for (const auto &trk : active_tracks_) {
    auto new_x_state = trk->predict();
    v_trcks.push_back(new_x_state);
  }

  // prepare veloctities for assoctiation
  Eigen::MatrixXf velocities(active_tracks_.size(), 2);
  for (int i = 0; i < active_tracks_.size(); i++) {
    auto trk = active_tracks_[i];
    if (trk->get_velocity().has_value())
      velocities.row(i) = trk->get_velocity().value().transpose();
    else
      velocities.row(i) = Eigen::RowVector2f{0, 0};
  }

  // last observation for each track
  std::vector<Eigen::Vector<float, MEAS_DIM + 1>> last_boxes;
  for (const auto &trk : active_tracks_)
    last_boxes.push_back(trk->get_last_observation());

  // last k_obs for each track
  std::vector<Eigen::Vector<float, MEAS_DIM + 1>> k_obs;
  for (const auto &trk : active_tracks_) {
    auto tmp = k_previous_obs(trk->get_observations(), trk->get_age(),
                              params_.delta_t);
    k_obs.push_back(std::move(tmp));
  }

  auto as_res = associate(dets, v_trcks, params_.det_thresh, velocities, k_obs,
                          params_.inertia);

  // update matched detections
  for (const auto &trk : as_res.matched) {
    active_tracks_[trk.second]->update(dets[trk.first].head(5),
                                       dets[trk.first][4], dets[trk.first][5],
                                       dets[trk.first][6]);
  }

  // Second ass with IoU only of left tracks
  if (!as_res.unmatched_dets.empty() && !as_res.unmatched_trks.empty()) {

    // form cost matrix
    int N_left = as_res.unmatched_dets.size();
    int M_left = as_res.unmatched_trks.size();

    Eigen::MatrixXf left_dets(N_left, 4);
    Eigen::MatrixXf left_trks(M_left, 4);

    for (int i = 0; i < N_left; ++i) {
      int det_idx = as_res.unmatched_dets[i];
      left_dets.row(i) = dets[det_idx].head(4);
    }
    for (int i = 0; i < M_left; ++i) {
      int trk_idx = as_res.unmatched_trks[i];
      left_trks.row(i) = last_boxes[trk_idx].head(4);
    }

    Eigen::MatrixXf iou_left = iou_batch(left_dets, left_trks);

    // std::cout << "iou_left:\n" << iou_left << std::endl;
    if (iou_left.maxCoeff() >= params_.iou_threshold) {
      std::vector<std::vector<double>> cost_left =
          eigen_to_vector2d((2.5f - iou_left.array()).matrix());
      std::vector<int> rematch_assignment;
      hung_algo.Solve(cost_left, rematch_assignment);

      std::vector<bool> keep_det(as_res.unmatched_dets.size(), true);
      std::vector<bool> keep_trk(as_res.unmatched_trks.size(), true);

      for (int i = 0; i < rematch_assignment.size(); ++i) {
        int t = rematch_assignment[i];
        if (t >= 0 && t < iou_left.cols() &&
            iou_left(i, t) >= params_.iou_threshold) {
          int det_idx = as_res.unmatched_dets[i];
          int trk_idx = as_res.unmatched_trks[t];

          active_tracks_[trk_idx]->update(dets[det_idx].head(5),
                                          dets[det_idx][4], dets[det_idx][5],
                                          dets[det_idx][6]);

          as_res.matched.emplace_back(det_idx, trk_idx);

          keep_det[i] = false;
          keep_trk[t] = false;
        }
      }

      std::vector<int> new_dets, new_trks;
      for (size_t i = 0; i < keep_det.size(); ++i)
        if (keep_det[i])
          new_dets.push_back(as_res.unmatched_dets[i]);
      for (size_t i = 0; i < keep_trk.size(); ++i)
        if (keep_trk[i])
          new_trks.push_back(as_res.unmatched_trks[i]);

      as_res.unmatched_dets = std::move(new_dets);
      as_res.unmatched_trks = std::move(new_trks);
    }
  }

  for (const auto &trk : as_res.unmatched_trks)
    active_tracks_[trk]->update(std::nullopt, -1, -1, -1);

  for (int det_idx : as_res.unmatched_dets) {
    auto trk = std::make_shared<KalmanBoxTracker>(
        dets[det_idx], dets[det_idx][5], dets[det_idx][6]);
    active_tracks_.push_back(trk);
  }

  std::vector<shared_ptr<KalmanBoxTracker>> outputs;
  for (auto it = active_tracks_.begin(); it != active_tracks_.end();) {
    auto &trk = *it;

    Eigen::Vector<float, 4> d;
    if (trk->get_last_observation().head(4).sum() < 0) {
      d = trk->get_state().head(4);
    } else {
      d = trk->get_last_observation().head(4);
    }

    // For MOT benchmark
    if (trk->get_time_since_update() < 1 &&
        (trk->get_hit_streak() >= params_.min_hits ||
         frame_count_ <= params_.min_hits)) {
      outputs.push_back(trk);
    }

    if (trk->get_time_since_update() > params_.max_age) {
      it = active_tracks_.erase(it);
    } else {
      ++it;
    }
  }

  return outputs;
}

oc_sort::OcSort::OcSort(const oc_sort::OcSort::Params &params)
    : params_(params), detector_(params.engine_path, ""), hung_algo({}) {}

oc_sort::OcSort::~OcSort() {}