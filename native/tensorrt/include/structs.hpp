#pragma once
#include <opencv4/opencv2/core.hpp>
#include <string>

#include "NvInfer.h"

namespace detection {

enum StatusCode {
  SUCCESS = 0,
  ERROR_DURING_INIT = 1,
  ERROR_DURING_LOAD_ENGINE = 2,
  ERROR_DURING_PARSE_TENSORS = 3,
  ERROR_DURING_SETTING_BUFFER = 4,
  ERROR_DURING_ENQUEUE = 5,
  ERROR_DURING_H2D = 6,
  ERROR_DURING_D2H = 7,
  ERROR_CUDA_MEMORY_ALLOC = 8,
  ERROR_NO_OUTPUT_TENSORS = 9,
  ERROR_CUDA_MEMORY_OPERATION = 10
};

static std::string get_message_error(StatusCode statusCode) {
  switch (statusCode) {
  case (SUCCESS):
    return "Success.";
  case (ERROR_DURING_INIT):
    return "Error during initialization of logger or runtime.";
  case (ERROR_DURING_LOAD_ENGINE):
    return "Error during load engine.";
  case (ERROR_DURING_PARSE_TENSORS):
    return "Error during parse tensors.";
  case (ERROR_DURING_SETTING_BUFFER):
    return "Error during setting buffer.";
  case (ERROR_DURING_ENQUEUE):
    return "Error during enqueue.";
  case (ERROR_DURING_H2D):
    return "Error during H2D operation.";
  case (ERROR_DURING_D2H):
    return "Error during D2H operation.";
  case (ERROR_CUDA_MEMORY_ALLOC):
    return "Error during cuda memory alloc";
  case (ERROR_NO_OUTPUT_TENSORS):
    return "Error no output tensors.";
  case (ERROR_CUDA_MEMORY_OPERATION):
    return "Error in cuda memory operation.";
  }
}

struct IOTensor {
  IOTensor(nvinfer1::Dims dims, std::string name, int index, int type = 4)
      : dims_(dims), name_(name), index_(index), type_(type) {}

  size_t get_element_count() const {
    size_t count = 1;
    for (int i = 0; i < dims_.nbDims; ++i) {
      count *= dims_.d[i];
    }
    return count;
  }

  size_t get_volume() const noexcept { return get_element_count() * type_; }

  nvinfer1::Dims dims_;
  std::string name_;
  int64_t index_;
  int64_t type_;
};

struct BoundingBox {
  int x{0};
  int y{0};
  int width{0};
  int height{0};

  BoundingBox() = default;
  BoundingBox(int _x, int _y, int w, int h)
      : x(_x), y(_y), width(w), height(h) {}

  float area() const { return static_cast<float>(width * height); }

  BoundingBox intersect(const BoundingBox &other) const {
    int xStart = std::max(x, other.x);
    int yStart = std::max(y, other.y);
    int xEnd = std::min(x + width, other.x + other.width);
    int yEnd = std::min(y + height, other.y + other.height);
    int iw = std::max(0, xEnd - xStart);
    int ih = std::max(0, yEnd - yStart);
    return BoundingBox(xStart, yStart, iw, ih);
  }
};

struct Detection {
  BoundingBox box;
  float conf{0.f};
  int classId{0};
};

} // namespace detection