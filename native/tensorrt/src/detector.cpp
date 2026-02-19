#include <cmath>
#include <cstdint>
#include <cstring>
#include <memory>
#include <opencv2/core/types.hpp>
#include <opencv4/opencv2/opencv.hpp>

#include "NvInferRuntime.h"
#include "NvInferRuntimeBase.h"
#include "detector.hpp"
#include "structs.hpp"
#include "utils.hpp"

using namespace detection;

Detector::Detector(const std::string &engine_path,
                   const std::string &file_class)
    : pt_(engine_path) {
  pt_.set_table_name("pre-processing,inference,post-processing\n");
#if TRT == 10
  tensorrt_base_ = std::make_unique<TensorRTv10Engine>(engine_path);
#else
  tensorrt_base_ = std::make_unique<TensorRTv8Engine>(engine_path);
#endif
  logger_ = tensorrt_base_->get_logger();
  auto status = tensorrt_base_->init();
  if (status != StatusCode::SUCCESS) {
    logger_->log(nvinfer1::ILogger::Severity::kERROR,
                 get_message_error(status).c_str());
    return;
  }
  status = tensorrt_base_->parse_tensors();
  if (status != StatusCode::SUCCESS) {
    logger_->log(nvinfer1::ILogger::Severity::kERROR,
                 get_message_error(status).c_str());
    return;
  }
  tensorrt_base_->allocate_buffers();
  tensorrt_base_->set_buffers();
}

cv::Mat Detector::preprocess(const cv::Mat &image,
                             std::vector<uint8_t> &blobPtr,
                             std::vector<int64_t> &inputTensorShape) noexcept {
  cv::Mat letterboxImage;

  cv::Size detector_input(inputTensorShape[2], inputTensorShape[3]);

  utils::letterBox(image, letterboxImage, detector_input,
                   cv::Scalar(114, 114, 114),
                   /*auto_=*/false,
                   /*scaleFill=*/false, /*scaleUp=*/true, /*stride=*/32);

  letterboxImage.convertTo(letterboxImage, CV_32FC3, 1.0f / 255.0f);
  size_t size = static_cast<size_t>(letterboxImage.rows) *
                static_cast<size_t>(letterboxImage.cols) * 3;
  blobPtr.resize(size * 4);
  std::vector<cv::Mat> channels(3);
  for (int c = 0; c < 3; ++c) {
    channels[c] = cv::Mat(letterboxImage.rows, letterboxImage.cols, CV_32FC1,
                          blobPtr.data() +
                              c * (letterboxImage.rows * letterboxImage.cols));
  }
  cv::split(letterboxImage, channels);

  return letterboxImage;
}

std::vector<detection::Detection>
Detector::detect(const cv::Mat &image, float confThreshold,
                 float iouThreshold) noexcept {
  std::unique_ptr<float[]> blobPtr;

  // input tensor
  pt_.start("detect");
  auto input_tensor = tensorrt_base_->input_[0];
  std::vector<int64_t> input_shape = {1, 3, input_tensor.dims_.d[3],
                                      input_tensor.dims_.d[2]};

  cv::Mat letterbox_img =
      preprocess(image, tensorrt_base_->host_input_buffers_[0], input_shape);
  pt_.stop("detect", ",");

  if (tensorrt_base_->copy_all_inputs_to_device()) {
    logger_->log(Logger::Severity::kERROR, "H2D failed");
    return {};
  }

  pt_.start("detect");
  if (tensorrt_base_->inference()) {
    logger_->log(Logger::Severity::kERROR, "Inference failed");
    return {};
  }
  pt_.stop("detect", ",");

  pt_.start("detect");
  if (tensorrt_base_->copy_all_outputs_to_host()) {
    logger_->log(Logger::Severity::kERROR, "D2H failed");
    return {};
  }
  cv::Size letterboxSize(static_cast<int>(input_tensor.dims_.d[3]),
                         static_cast<int>(input_tensor.dims_.d[2]));

  auto result = postprocess(image.size(), letterboxSize,
                            tensorrt_base_->host_output_buffers_, confThreshold,
                            iouThreshold);
  pt_.stop("detect", "\n");
  return result;
}

std::vector<detection::Detection>
Detector::postprocess(const cv::Size &origSize, const cv::Size &letterboxSize,
                      const std::vector<std::vector<uint8_t>> &outputs,
                      float confThreshold, float iouThreshold) noexcept {
  std::vector<detection::Detection> results;

  bool yolov8 = false;
  if (outputs.size() == 1)
    yolov8 = true;
  // Extract outputs
  const float *output0_ptr;
  const float *output1_ptr;
  if (yolov8) {
    output0_ptr = reinterpret_cast<const float *>(outputs[0].data());
  } else {
    output0_ptr = reinterpret_cast<const float *>(outputs[0].data());
    output1_ptr = reinterpret_cast<const float *>(outputs[1].data());
  }

  // Get shapes
#ifdef TRT

#if TRT == 10
  nvinfer1::Dims dims0, dims1;
  if (yolov8) {
    dims0 = tensorrt_base_->output_[0].dims_;
  } else {
    dims0 = tensorrt_base_->output_[0].dims_;
    dims1 = tensorrt_base_->output_[1].dims_;
  }
#else
  auto &dims0 = tensorrt_base_->output_[1].dims_;
  auto &dims1 = tensorrt_base_->output_[0].dims_;
#endif

#endif
  std::vector<int64_t> shape0(dims0.d, dims0.d + dims0.nbDims);

  size_t num_features = shape0[1];
  size_t num_detections = shape0[2];
  if (!yolov8) {
    num_features = shape0[2] + dims1.d[2];
    num_detections = shape0[1];
  }
  // Early exit if no detections
  if (num_detections == 0) {
    return results;
  }

  const int numClasses =
      static_cast<int>(num_features - 4); // Corrected number of classes

  // Validate numClasses
  if (numClasses <= 0) {
    throw std::runtime_error("Invalid number of classes.");
  }

  const int numBoxes = static_cast<int>(num_detections);

  // Constants from model architecture
  constexpr int BOX_OFFSET = 0;
  int CLASS_CONF_OFFSET = 0;
  if (yolov8)
    CLASS_CONF_OFFSET = 4;
  // 1. Process detections
  std::vector<detection::BoundingBox> boxes;
  boxes.reserve(numBoxes);
  std::vector<float> confidences;
  confidences.reserve(numBoxes);
  std::vector<int> classIds;
  classIds.reserve(numBoxes);
  std::vector<std::vector<float>> maskCoefficientsList;
  maskCoefficientsList.reserve(numBoxes);

  for (int i = 0; i < numBoxes; ++i) {
    // Extract box coordinates
    float xc, yc, w, h;
    if (yolov8) {
      xc = output0_ptr[BOX_OFFSET * numBoxes + i];
      yc = output0_ptr[(BOX_OFFSET + 1) * numBoxes + i];
      w = output0_ptr[(BOX_OFFSET + 2) * numBoxes + i];
      h = output0_ptr[(BOX_OFFSET + 3) * numBoxes + i];
    } else {
      xc = output0_ptr[i * 4];
      yc = output0_ptr[i * 4 + 1];
      w = output0_ptr[i * 4 + 2];
      h = output0_ptr[i * 4 + 3];
    }

    // Convert to xyxy format
    detection::BoundingBox box;
    if (yolov8) {
      box = {static_cast<int>(std::round(xc - w / 2.0f)),
             static_cast<int>(std::round(yc - h / 2.0f)),
             static_cast<int>(std::round(w)), static_cast<int>(std::round(h))};
    } else {
      box = {static_cast<int>(std::round(xc)), static_cast<int>(std::round(yc)),
             static_cast<int>(std::round(w - xc)),
             static_cast<int>(std::round(h - yc))};
    }

    // Get class confidence
    float maxConf = 0.0f;
    int classId = -1;
    for (int c = 0; c < numClasses; ++c) {
      float conf = yolov8 ? output0_ptr[(CLASS_CONF_OFFSET + c) * numBoxes + i]
                          : output1_ptr[i * numClasses + c];
      if (conf > maxConf) {
        maxConf = conf;
        classId = c;
      }
    }

    if (maxConf < confThreshold)
      continue;

    // Store detection
    boxes.push_back(box);
    confidences.push_back(maxConf);
    classIds.push_back(classId);
  }

  // Early exit if no boxes after confidence threshold
  if (boxes.empty()) {
    return results;
  }

  // 2. Apply NMS
  std::vector<int> nmsIndices;
  utils::NMSBoxes(boxes, confidences, confThreshold, iouThreshold, nmsIndices);

  if (nmsIndices.empty()) {
    return results;
  }

  // 3. Prepare final results
  results.reserve(nmsIndices.size());

  for (const int idx : nmsIndices) {
    Detection seg;
    seg.box = boxes[idx];
    seg.conf = confidences[idx];
    seg.classId = classIds[idx];

    // 4. Scale box to original image
    seg.box = utils::scaleCoords(letterboxSize, seg.box, origSize, true);
    results.push_back(seg);
  }

  return results;
};
