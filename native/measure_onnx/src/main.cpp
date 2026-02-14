#include <vector>
#include <getopt.h>
#include <cxxopts.hpp>

#include <opencv2/opencv.hpp>

#include "inference.h"

using namespace std;
using namespace cv;


int main(int argc, char **argv)
{
    cxxopts::Options options("measure_onnx", "measure speed of yolo models with onnx runtime");
    options.add_options()
        ("i,image", "Image path", cxxopts::value<std::string>()->default_value("/home/matvey/projects/jmot/bus.jpg"))
        ("m,model", "Model path", cxxopts::value<std::string>()->default_value("/home/matvey/projects/jmot/scripts/models/yolo_nas_s.onnx"))
        ("t,times", "Amount of measures", cxxopts::value<int>()->default_value("10"))
    ;

    auto result = options.parse(argc, argv);

    bool runOnGPU = true;

    Inference inf(result["model"].as<std::string>(), cv::Size(640, 640), "classes.txt", runOnGPU);

    std::vector<std::string> imageNames;
    imageNames.push_back(result["image"].as<std::string>());
    // imageNames.push_back(projectBasePath + "/zidane.jpg");

    for (int j = 0; j < result["times"].as<int>(); j++) {
        for (int i = 0; i < imageNames.size(); ++i)
        {
            cv::Mat frame = cv::imread(imageNames[i]);

            std::vector<Detection> output = inf.runInference(frame);

            int detections = output.size();

            for (int i = 0; i < detections; ++i)
            {
                Detection detection = output[i];

                cv::Rect box = detection.box;
                cv::Scalar color = detection.color;

                // Detection box
                cv::rectangle(frame, box, color, 2);

                // Detection box text
                std::string classString = detection.className + ' ' + std::to_string(detection.confidence).substr(0, 4);
                cv::Size textSize = cv::getTextSize(classString, cv::FONT_HERSHEY_DUPLEX, 1, 2, 0);
                cv::Rect textBox(box.x, box.y - 40, textSize.width + 10, textSize.height + 20);

                cv::rectangle(frame, textBox, color, cv::FILLED);
                cv::putText(frame, classString, cv::Point(box.x + 5, box.y - 10), cv::FONT_HERSHEY_DUPLEX, 1, cv::Scalar(0, 0, 0), 2, 0);
            }

            // float scale = 0.8;
            // cv::resize(frame, frame, cv::Size(frame.cols*scale, frame.rows*scale));
            // cv::imshow("Inference", frame);
            // cv::waitKey(-1);
        }
    }
}
