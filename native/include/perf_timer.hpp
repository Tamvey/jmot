#include <chrono>
#include <fstream>
#include <iostream>

class perf_timer {
public:
  perf_timer(const std::string &model_path) {
    auto last_slash = model_path.find_last_of("/");
    out_.open(model_path.substr(last_slash + 1));
  }

  void set_table_name(const std::string &table_name) { out_ << table_name; }

  void start() { last_ = std::chrono::high_resolution_clock::now(); }

  void stop(const std::string &postfix) {
    auto spent = std::chrono::high_resolution_clock::now() - last_;
    out_ << std::chrono::duration_cast<std::chrono::milliseconds>(spent).count()
         << postfix;
  }

private:
  std::chrono::high_resolution_clock::time_point last_;
  std::ofstream out_;
};