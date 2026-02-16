#pragma once
#include <exception>
#include <fstream>
#include <memory>
#include <vector>

#include "logger.hpp"
#include "structs.hpp"

namespace detection {

class TensorRTBase {
public:
  TensorRTBase(const std::string &engine_path);
  virtual ~TensorRTBase() = 0;

  virtual StatusCode init() noexcept;
  virtual StatusCode load_engine_from_file() noexcept;
  virtual StatusCode read_file(std::vector<char> &dest) const noexcept;

  virtual StatusCode allocate_buffers() noexcept;
  virtual StatusCode allocate_host_buffers() noexcept;
  virtual StatusCode allocate_device_buffers() noexcept;

  virtual StatusCode copyH2D(size_t input_index = 0) noexcept;
  virtual StatusCode copyD2H(size_t output_index = 0) noexcept;
  virtual StatusCode copy_all_inputs_to_device() noexcept;
  virtual StatusCode copy_all_outputs_to_host() noexcept;
  virtual StatusCode cleanup_device_memory() noexcept;

  virtual StatusCode parse_tensors() noexcept = 0;
  virtual StatusCode inference() = 0;
  virtual StatusCode set_buffers() = 0;

  virtual std::shared_ptr<Logger> get_logger() const noexcept;

public:
  std::shared_ptr<Logger> logger_;
  std::unique_ptr<nvinfer1::IRuntime> runtime_;
  std::unique_ptr<nvinfer1::ICudaEngine> engine_;
  std::unique_ptr<nvinfer1::IExecutionContext> context_;

  std::vector<IOTensor> input_;
  std::vector<IOTensor> output_;

  std::vector<std::vector<uint8_t>> host_input_buffers_;
  std::vector<std::vector<uint8_t>> host_output_buffers_;

  std::vector<void *> device_input_buffers_;
  std::vector<void *> device_output_buffers_;

  std::string engine_path_;
  cudaStream_t stream_;
};

// declaration of derived classes from base
class TensorRTv8Engine : public TensorRTBase {
public:
  TensorRTv8Engine(const std::string &engine_path);

  StatusCode set_buffers() noexcept override;
  StatusCode inference() noexcept override;
  StatusCode parse_tensors() noexcept override;
};

class TensorRTv10Engine : public TensorRTBase {
public:
  TensorRTv10Engine(const std::string &engine_path);

  StatusCode set_buffers() noexcept override;
  StatusCode inference() noexcept override;
  StatusCode parse_tensors() noexcept override;
};

} // namespace detection