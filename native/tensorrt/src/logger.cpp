#include <iostream>

#include "NvInferRuntime.h"
#include "logger.hpp"

void detection::Logger::log(Severity severity, const char *msg) noexcept {
  std::printf("%s | %s\n", level_to_string(severity).c_str(), msg);
}

std::string detection::Logger::level_to_string(Severity severity) noexcept {
  switch (severity) {
  case Severity::kINFO:
    return "INFO";
  case Severity::kWARNING:
    return "WARNING";
  case Severity::kERROR:
  case Severity::kINTERNAL_ERROR:
    return "ERROR";
  case Severity::kVERBOSE:
    return "VERBOSE";
  default:
    return "UNKNOWN";
  }
}