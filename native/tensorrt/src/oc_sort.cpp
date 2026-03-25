#include <vector>

#include "oc_sort.hpp"

std::vector<oc_sort::TrackOutput>
oc_sort::OcSort::update(const std::vector<Eigen::Vector<float, 6>> &dets,
                        const cv::Mat &img) {
  frame_count_++;
  auto h = img.rows;
  auto w = img.cols;
  return {};
}

oc_sort::OcSort::OcSort(const oc_sort::OcSort::Params &params)
    : params_(params), detector_(params.engine_path, "") {}

oc_sort::OcSort::~OcSort() {}