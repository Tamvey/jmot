#pragma once
#include <memory>

#include "opencv2/opencv.hpp"
#include "perf_timer.hpp"
#include "structs.hpp"
#include "tensorrt_base.hpp"

namespace detection {

class Detector {
public:
  Detector(const std::string &engine_path, const std::string &file_class);

  std::vector<Detection> detect(const cv::Mat &image, float confThreshold,
                                float iouThreshold) noexcept;
  cv::Mat preprocess(const cv::Mat &image, std::unique_ptr<float[]> &blobPtr,
                     std::vector<int64_t> &inputTensorShape) noexcept;
  std::vector<Detection>
  postprocess(const cv::Size &origSize, const cv::Size &letterboxSize,
              const std::vector<std::vector<uint8_t>> &outputs,
              float confThreshold, float iouThreshold) noexcept;

private:
  std::unique_ptr<TensorRTBase> tensorrt_base_;
  std::shared_ptr<Logger> logger_;
  perf_timer pt_;
};

} // namespace detection
