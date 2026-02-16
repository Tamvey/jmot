#include <fstream>
#include <opencv4/opencv2/opencv.hpp>
#include <random>
#include <vector>

#include "structs.hpp"

namespace utils {

std::vector<std::string> getClassNames(const std::string &path);

std::vector<cv::Scalar>
generateColors(const std::vector<std::string> &classNames, int seed = 42);

void NMSBoxes(const std::vector<detection::BoundingBox> &boxes,
              const std::vector<float> &scores, float scoreThreshold,
              float nmsThreshold, std::vector<int> &indices);

detection::BoundingBox scaleCoords(const cv::Size &letterboxShape,
                                   const detection::BoundingBox &coords,
                                   const cv::Size &originalShape,
                                   bool p_Clip = true);
cv::Mat sigmoid(const cv::Mat &src);

void letterBox(const cv::Mat &image, cv::Mat &outImage,
               const cv::Size &newShape,
               const cv::Scalar &color = cv::Scalar(114, 114, 114),
               bool auto_ = true, bool scaleFill = false, bool scaleUp = true,
               int stride = 32);
} // namespace utils
