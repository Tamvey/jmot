#pragma once
#include <memory>

#include "perf_timer.hpp"
#include "structs.hpp"
#include "tensorrt_base.hpp"

namespace detection {

class Detector {
public:
  Detector(const std::string &engine_path,
           const SAHIParams &params = SAHIParams(640, 640, 0.2, 0.2),
           float conf_thresh = 0.3, float iou_thresh = 0.3);

  std::vector<Detection> detect(const cv::Mat &image, bool useSahi) noexcept;
  void preprocess(const cv::Mat &image, std::vector<uint8_t> &host_input_buffer,
                  IOTensor &inputTensorShape,
                  const std::vector<cv::Rect> &image_slices) noexcept;
  std::vector<Detection>
  postprocess(const cv::Size &origSize, const cv::Size &letterboxSize,
              const std::vector<std::vector<uint8_t>> &outputs,
              const std::vector<cv::Rect> &image_slices, float confThreshold,
              float iouThreshold) noexcept;

private:
  std::unique_ptr<TensorRTBase> tensorrt_base_;
  std::shared_ptr<Logger> logger_;
  perf_timer pt_;
  SAHIParams params_;
  float conf_thresh_, iou_thresh_;
};

} // namespace detection
