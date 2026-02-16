#include "tensorrt_base.hpp"
#include "NvInferRuntime.h"
#include "structs.hpp"
#include <cuda_runtime_api.h>
#include <driver_types.h>
#include <iostream>
#include <memory>
#include <utility>
#include <vector>

using namespace detection;

TensorRTBase::TensorRTBase(const std::string &engine_path)
    : engine_path_(engine_path), device_input_buffers_(),
      device_output_buffers_(), host_input_buffers_(), host_output_buffers_(),
      stream_(nullptr) {}

TensorRTBase::~TensorRTBase() { cleanup_device_memory(); }

StatusCode TensorRTBase::init() noexcept {
  try {
    logger_ = std::make_shared<Logger>();
  } catch (std::exception &e) {
    return StatusCode::ERROR_DURING_INIT;
  }

  runtime_ = std::unique_ptr<nvinfer1::IRuntime>(
      nvinfer1::createInferRuntime(*this->logger_));
  if (!runtime_)
    return StatusCode::ERROR_DURING_INIT;

  // load engine file
  auto status = load_engine_from_file();
  if (status != StatusCode::SUCCESS) {
    logger_->log(nvinfer1::ILogger::Severity::kERROR,
                 get_message_error(status).c_str());
    return StatusCode::ERROR_DURING_INIT;
  }
  if (!engine_)
    return StatusCode::ERROR_DURING_INIT;

  // create execution context based on engine
  context_ = std::unique_ptr<nvinfer1::IExecutionContext>(
      engine_->createExecutionContext());

  if (!context_)
    return StatusCode::ERROR_DURING_INIT;

  return StatusCode::SUCCESS;
}

StatusCode TensorRTBase::load_engine_from_file() noexcept {

  std::vector<char> model_data;

  // read binary engine file
  auto read_result = read_file(model_data);

  if (read_result != StatusCode::SUCCESS)
    return read_result;

  engine_ = std::unique_ptr<nvinfer1::ICudaEngine>(
      runtime_->deserializeCudaEngine(model_data.data(), model_data.size()));
  if (!engine_)
    return StatusCode::ERROR_DURING_LOAD_ENGINE;

  return StatusCode::SUCCESS;
};

StatusCode TensorRTBase::read_file(std::vector<char> &dest) const noexcept {
  try {
    std::ifstream in(engine_path_, std::ios::binary);

    // define model size
    in.seekg(0, std::ios::end);
    size_t size = in.tellg();
    in.seekg(0, std::ios::beg);

    dest.resize(size);

    in.read(dest.data(), size);

    return StatusCode::SUCCESS;
  } catch (std::exception &e) {
    return StatusCode::ERROR_DURING_LOAD_ENGINE;
  }
};

std::shared_ptr<detection::Logger> TensorRTBase::get_logger() const noexcept {
  return logger_;
}

StatusCode TensorRTBase::allocate_buffers() noexcept {
  allocate_host_buffers();
  allocate_device_buffers();
  return StatusCode::SUCCESS;
}

StatusCode TensorRTBase::allocate_host_buffers() noexcept {
  for (const auto &tensor : input_) {
    host_input_buffers_.emplace_back(tensor.get_volume());
  }
  for (const auto &tensor : output_) {
    host_output_buffers_.emplace_back(tensor.get_volume());
  }
  return StatusCode::SUCCESS;
}

StatusCode TensorRTBase::allocate_device_buffers() noexcept {
  for (const auto &tensor : input_) {
    void *d_ptr;
    cudaMalloc(&d_ptr, tensor.get_volume());
    device_input_buffers_.push_back(d_ptr);
  }
  for (const auto &tensor : output_) {
    void *d_ptr;
    cudaMalloc(&d_ptr, tensor.get_volume());
    device_output_buffers_.push_back(d_ptr);
  }
  if (!stream_) {
    cudaStreamCreate(&stream_);
  }
  return StatusCode::SUCCESS;
}

StatusCode TensorRTBase::copyH2D(size_t input_index) noexcept {
  cudaMemcpyAsync(device_input_buffers_[input_index],
                  host_input_buffers_[input_index].data(),
                  host_input_buffers_[input_index].size(),
                  cudaMemcpyHostToDevice, stream_);
  return StatusCode::SUCCESS;
}

StatusCode TensorRTBase::copyD2H(size_t output_index) noexcept {
  cudaMemcpyAsync(host_output_buffers_[output_index].data(),
                  device_output_buffers_[output_index],
                  host_output_buffers_[output_index].size(),
                  cudaMemcpyDeviceToHost, stream_);
  cudaStreamSynchronize(stream_);
  return StatusCode::SUCCESS;
}

StatusCode TensorRTBase::copy_all_inputs_to_device() noexcept {
  for (size_t i = 0; i < host_input_buffers_.size(); ++i) {
    copyH2D(i);
  }
  return StatusCode::SUCCESS;
}

StatusCode TensorRTBase::copy_all_outputs_to_host() noexcept {
  for (size_t i = 0; i < host_output_buffers_.size(); ++i) {
    copyD2H(i);
  }
  return StatusCode::SUCCESS;
}

StatusCode TensorRTBase::cleanup_device_memory() noexcept {
  for (void *ptr : device_input_buffers_) {
    if (ptr)
      cudaFree(ptr);
  }
  device_input_buffers_.clear();

  for (void *ptr : device_output_buffers_) {
    if (ptr)
      cudaFree(ptr);
  }
  device_output_buffers_.clear();

  if (stream_) {
    cudaStreamDestroy(stream_);
    stream_ = nullptr;
  }
  return StatusCode::SUCCESS;
}
