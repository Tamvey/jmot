#include "tensorrt_base.hpp"

using namespace detection;

TensorRTv8Engine::TensorRTv8Engine(const std::string &engine_path)
    : TensorRTBase(engine_path) {};

StatusCode TensorRTv8Engine::parse_tensors() noexcept {
  auto amount = engine_->getNbBindings();

  if (amount == 0)
    return StatusCode::ERROR_DURING_PARSE_TENSORS;

  for (auto i = 0; i < amount; i++) {
    auto index = i;
    std::string name = engine_->getBindingName(index);

    auto tensor = IOTensor(engine_->getBindingDimensions(index), name, index);

    if (name.find("output") != std::string::npos) {
      output_.push_back(std::move(tensor));
    } else {
      input_.push_back(std::move(tensor));
    }
  }
  return StatusCode::SUCCESS;
};

StatusCode TensorRTv8Engine::set_buffers() noexcept {
  return StatusCode::SUCCESS;
};

StatusCode TensorRTv8Engine::inference() noexcept {
  const size_t num_bindings = 3;
  void *const bindings_[num_bindings] = {device_input_buffers_[0],
                                         device_output_buffers_[1],
                                         device_output_buffers_[0]};
  void *const *bindings = bindings_;
  if (!context_->enqueueV2(&bindings_[0], stream_, nullptr))
    return StatusCode::ERROR_DURING_ENQUEUE;
  cudaStreamSynchronize(stream_);
  return StatusCode::SUCCESS;
};