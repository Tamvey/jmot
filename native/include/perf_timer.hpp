#include <chrono>
#include <fstream>
#include <iostream>
#include <unordered_map>

class perf_timer {
public:
  perf_timer(const std::string &model_path) {
    auto last_slash = model_path.find_last_of("/");
    out_.open(model_path.substr(last_slash + 1));
  }

  void set_table_name(const std::string &table_name) { out_ << table_name; }

  void start(const std::string &fname) {
    auto now = std::chrono::high_resolution_clock::now();
    if (tps.find(fname) != tps.end()) {
      tps[fname] = now;
    } else {
      tps.insert({fname, now});
    }
  }

  void stop(const std::string &fname, const std::string &postfix) {
    if (tps.find(fname) == tps.end())
      return;
    auto spent =
        std::chrono::high_resolution_clock::now() - tps.find(fname)->second;
    out_ << std::chrono::duration_cast<std::chrono::milliseconds>(spent).count()
         << postfix;
  }

private:
  std::ofstream out_;
  std::unordered_map<std::string,
                     std::chrono::high_resolution_clock::time_point>
      tps;
};