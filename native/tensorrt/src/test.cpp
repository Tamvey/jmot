#include <cxxopts.hpp>
#include <getopt.h>
#include <string>
#include <vector>

#include <opencv2/opencv.hpp>

#include "detector.hpp"
#include "structs.hpp"

using namespace std;
using namespace cv;

int main(int argc, char **argv) {
  cxxopts::Options options("tensorrt",
                           "measure speed of yolo models with tensorrt");
  options.add_options()(
      "i,image", "Image path",
      cxxopts::value<std::string>()->default_value(
          "/home/matvey/projects/jmot/urban-view-with-cars-street.jpg"))(
      "m,model", "Model path",
      cxxopts::value<std::string>()->default_value(
          "/home/matvey/projects/jmot/scripts/models/yolo11l.engine"))(
      "t,times", "Amount of measures",
      cxxopts::value<int>()->default_value("10"));

  auto result = options.parse(argc, argv);

  auto detector = detection::Detector(result["model"].as<std::string>(), "");

  std::vector<std::string> imageNames;
  imageNames.push_back(result["image"].as<std::string>());

  for (int j = 0; j < result["times"].as<int>(); j++) {
    for (int i = 0; i < imageNames.size(); ++i) {
      cv::Mat frame = cv::imread(imageNames[i]);

      std::vector<detection::Detection> output =
          detector.detect(frame, 0.25, 0.45);
#ifdef REFLECT
      int detections = output.size();

      for (int i = 0; i < detections; ++i) {
        auto detection = output[i];

        detection::BoundingBox box = detection.box;
        cv::Scalar color = cv::Scalar(255, 0, 0);

        // Detection box
        cv::rectangle(frame, {box.x, box.y, box.width, box.height}, color, 2);

        // Detection box text
        std::string classString = std::to_string(detection.classId) + ' ' +
                                  std::to_string(detection.conf).substr(0, 4);
        cv::Size textSize =
            cv::getTextSize(classString, cv::FONT_HERSHEY_DUPLEX, 1, 2, 0);
        cv::Rect textBox(box.x, box.y - 40, textSize.width + 10,
                         textSize.height + 20);

        cv::rectangle(frame, textBox, color, cv::FILLED);
        cv::putText(frame, classString, cv::Point(box.x + 5, box.y - 10),
                    cv::FONT_HERSHEY_DUPLEX, 1, cv::Scalar(0, 0, 0), 2, 0);
      }

      float scale = 0.2;
      cv::resize(frame, frame,
                 cv::Size(frame.cols * scale, frame.rows * scale));
      cv::imshow("Inference", frame);
      cv::waitKey(-1);
#endif
    }
  }
}
