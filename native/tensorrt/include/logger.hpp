#include <string>

#include "NvInfer.h"

namespace detection {

class Logger : public nvinfer1::ILogger {
public:
  void log(Severity severity, const char *msg) noexcept override;
  std::string level_to_string(Severity severity) noexcept;
};

} // namespace detection
