#include <cmath>
#include <cstdint>
#include <cstring>
#include <future>
#include <memory>
#include <opencv2/core/types.hpp>
#include <opencv4/opencv2/opencv.hpp>
#include <vector>

#include "NvInferRuntime.h"
#include "NvInferRuntimeBase.h"
#include "detector.hpp"
#include "structs.hpp"
#include "utils.hpp"

using namespace detection;

Detector::Detector(const std::string &engine_path, const SAHIParams &params,
                   float conf_thresh, float iou_thresh)
    : pt_(engine_path), params_(params), conf_thresh_(conf_thresh),
      iou_thresh_(iou_thresh) {
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

void Detector::preprocess(const cv::Mat &image,
                          std::vector<uint8_t> &host_input_buffer,
                          IOTensor &inputTensorShape,
                          const std::vector<cv::Rect> &image_slices) noexcept {
  // prepare buffer for batch size inputTensorShape[0]
  size_t size = inputTensorShape.get_volume();
  host_input_buffer.resize(size);

  cv::Size detector_input(inputTensorShape.dims_.d[2],
                          inputTensorShape.dims_.d[3]);
  std::vector<std::future<void>> preprocess_futures;
  // fulfill all batch
  for (auto ptr_i = 0; ptr_i < inputTensorShape.dims_.d[0]; ptr_i++) {
    preprocess_futures.push_back(std::async(std::launch::async, [&, ptr_i] {
      // stop if slices more than place in batch
      if (image_slices.size() == ptr_i)
        return;

      cv::Mat letterboxImage;
      std::vector<cv::Mat> channels(3);

      detection::utils::letterBox(image(image_slices[ptr_i]), letterboxImage,
                                  detector_input, cv::Scalar(114, 114, 114),
                                  /*auto_=*/false,
                                  /*scaleFill=*/false, /*scaleUp=*/true,
                                  /*stride=*/32);

      letterboxImage.convertTo(letterboxImage, CV_32FC3, 1.0f / 255.0f);

      for (int c = 0; c < 3; ++c) {
        auto offset_from_batch =
            ptr_i * (letterboxImage.rows * letterboxImage.cols) * 3 * 4;
        channels[c] =
            cv::Mat(letterboxImage.rows, letterboxImage.cols, CV_32FC1,
                    host_input_buffer.data() + offset_from_batch +
                        c * (letterboxImage.rows * letterboxImage.cols) * 4);
      }
      cv::split(letterboxImage, channels);
    }));
  }
  for (auto &fut : preprocess_futures)
    fut.wait();
}

std::vector<detection::Detection> Detector::detect(const cv::Mat &image,
                                                   bool useSahi) noexcept {
  // input tensor
  pt_.start("detect");

  auto input_tensor = tensorrt_base_->input_[0];
  std::vector<cv::Rect> image_slices;

  if (useSahi)
    image_slices = detection::utils::slice_image(image.size(), params_);
  else
    image_slices = detection::utils::slice_image(
        image.size(),
        SAHIParams(image.size().height, image.size().width, 0, 0));

  preprocess(image, tensorrt_base_->host_input_buffers_[0], input_tensor,
             image_slices);

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
                            tensorrt_base_->host_output_buffers_, image_slices,
                            conf_thresh_, iou_thresh_);
  pt_.stop("detect", "\n");
  return result;
}

std::vector<detection::Detection>
Detector::postprocess(const cv::Size &origSize, const cv::Size &letterboxSize,
                      const std::vector<std::vector<uint8_t>> &outputs,
                      const std::vector<cv::Rect> &image_slices,
                      float confThreshold, float iouThreshold) noexcept {
  std::vector<detection::Detection> results;

  bool yolov8 = false;
  if (outputs.size() == 1)
    yolov8 = true;

  // extract outputs
  const float *output0_ptr;
  const float *output1_ptr;
  if (yolov8) {
    output0_ptr = reinterpret_cast<const float *>(outputs[0].data());
  } else {
    output0_ptr = reinterpret_cast<const float *>(outputs[0].data());
    output1_ptr = reinterpret_cast<const float *>(outputs[1].data());
  }

  // get shapes
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

  if (num_detections == 0) {
    return results;
  }

  const int numClasses =
      static_cast<int>(num_features - 4); // corrected number of classes

  // validate numClasses
  if (numClasses <= 0) {
    throw std::runtime_error("Invalid number of classes.");
  }

  const int numBoxes = static_cast<int>(num_detections);

  // constants from model architecture
  constexpr int BOX_OFFSET = 0;
  int CLASS_CONF_OFFSET = 0;
  if (yolov8)
    CLASS_CONF_OFFSET = 4;

  struct BatchResult {
    std::vector<Detection> detections;
  };

  std::vector<std::future<BatchResult>> futures;
  // process batch
  for (int ptr_i = 0; ptr_i < shape0[0]; ptr_i++) {
    futures.push_back(
        std::async(std::launch::async, [&, ptr_i, image_slices] -> BatchResult {
          BatchResult result;

          std::vector<detection::BoundingBox> boxes;
          std::vector<float> confidences;
          std::vector<int> classIds;

          boxes.reserve(numBoxes);
          confidences.reserve(numBoxes);
          classIds.reserve(numBoxes);

          std::vector<std::vector<float>> maskCoefficientsList;
          maskCoefficientsList.reserve(numBoxes);

          int offset = ptr_i * shape0[1] * shape0[2];

          for (int i = 0; i < numBoxes; ++i) {
            // extract box coordinates
            float xc, yc, w, h;
            if (yolov8) {
              xc = output0_ptr[offset + BOX_OFFSET * numBoxes + i];
              yc = output0_ptr[offset + (BOX_OFFSET + 1) * numBoxes + i];
              w = output0_ptr[offset + (BOX_OFFSET + 2) * numBoxes + i];
              h = output0_ptr[offset + (BOX_OFFSET + 3) * numBoxes + i];
            } else {
              xc = output0_ptr[i * 4];
              yc = output0_ptr[i * 4 + 1];
              w = output0_ptr[i * 4 + 2];
              h = output0_ptr[i * 4 + 3];
            }

            // convert to xyxy format
            detection::BoundingBox box;
            if (yolov8) {
              box = {static_cast<int>(std::round(xc - w / 2.0f)),
                     static_cast<int>(std::round(yc - h / 2.0f)),
                     static_cast<int>(std::round(w)),
                     static_cast<int>(std::round(h))};
            } else {
              box = {static_cast<int>(std::round(xc)),
                     static_cast<int>(std::round(yc)),
                     static_cast<int>(std::round(w - xc)),
                     static_cast<int>(std::round(h - yc))};
            }

            // get class confidence
            float maxConf = 0.0f;
            int classId = -1;
            for (int c = 0; c < numClasses; ++c) {
              float conf =
                  yolov8 ? output0_ptr[offset +
                                       (CLASS_CONF_OFFSET + c) * numBoxes + i]
                         : output1_ptr[i * numClasses + c];
              if (conf > maxConf) {
                maxConf = conf;
                classId = c;
              }
            }

            if (maxConf < confThreshold)
              continue;

            boxes.push_back(box);
            confidences.push_back(maxConf);
            classIds.push_back(classId);
          }

          if (boxes.empty()) {
            return result;
          }

          std::vector<int> nmsIndices;
          detection::utils::NMSBoxes(boxes, confidences, confThreshold,
                                     iouThreshold, nmsIndices);

          if (nmsIndices.empty()) {
            return result;
          }

          for (const int idx : nmsIndices) {
            Detection seg;
            seg.box = boxes[idx];
            seg.conf = confidences[idx];
            seg.classId = classIds[idx];

            seg.box = detection::utils::scaleCoords(
                letterboxSize, seg.box, image_slices[ptr_i].size(), true);
            seg.box.x += image_slices[ptr_i].x;
            seg.box.y += image_slices[ptr_i].y;

            boxes[idx].x += image_slices[ptr_i].x;
            boxes[idx].y += image_slices[ptr_i].y;

            result.detections.push_back(seg);
          }
          return result;
        }));
  }

  for (auto &fut : futures) {
    auto res = fut.get();
    for (auto det : res.detections)
      results.push_back(det);
  }
  // final nms above detections from all slices
  int size = results.size();

  std::vector<detection::BoundingBox> boxes;
  boxes.reserve(size);

  std::vector<float> confidences;
  confidences.reserve(size);

  for (int i = 0; i < size; i++) {
    boxes.push_back(results[i].box);
    confidences.push_back(results[i].conf);
  }

  std::vector<int> nmsIndices;
  detection::utils::NMSBoxes(boxes, confidences, confThreshold, iouThreshold,
                             nmsIndices);

  if (nmsIndices.empty()) {
    return results;
  }

  std::vector<Detection> final_results;
  final_results.reserve(nmsIndices.size());

  for (const int idx : nmsIndices) {
    Detection seg;
    seg.box = boxes[idx];
    seg.conf = confidences[idx];
    seg.classId = results[idx].classId;

    final_results.push_back(seg);
  }

  return final_results;
};
