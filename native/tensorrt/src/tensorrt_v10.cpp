#include "tensorrt_base.hpp"

using namespace detection;

TensorRTv10Engine::TensorRTv10Engine(const std::string &engine_path)
    : TensorRTBase(engine_path) {};

StatusCode TensorRTv10Engine::parse_tensors() noexcept {
  auto amount = engine_->getNbIOTensors();

  if (amount == 0)
    return StatusCode::ERROR_DURING_PARSE_TENSORS;

  for (auto i = 0; i < amount; i++) {
    auto index = i;
    std::string name = engine_->getIOTensorName(index);

    auto tensor = IOTensor(engine_->getTensorShape(name.c_str()), name, index);

    if (name.find("output") != std::string::npos) {
      output_.push_back(std::move(tensor));
    } else {
      input_.push_back(std::move(tensor));
    }
  }
  return StatusCode::SUCCESS;
};

StatusCode TensorRTv10Engine::set_buffers() noexcept {
  for (auto i = 0; i < device_input_buffers_.size(); i++) {
    context_->setTensorAddress(input_[i].name_.c_str(),
                               device_input_buffers_[i]);
  }

  for (auto i = 0; i < device_output_buffers_.size(); i++) {
    context_->setTensorAddress(output_[i].name_.c_str(),
                               device_output_buffers_[i]);
  }
  return StatusCode::SUCCESS;
};

StatusCode TensorRTv10Engine::inference() noexcept {
  if (!context_->enqueueV3(stream_))
    return StatusCode::ERROR_DURING_ENQUEUE;
  cudaStreamSynchronize(stream_);
  return StatusCode::SUCCESS;
};