#include "oc_sort.hpp"

int main() {
  oc_sort::OcSort::Params params = {};
  auto oc_sort = oc_sort::OcSort(params);

  cv::VideoCapture vc(
      "/media/matvey/EB6B-E36F/diploma/kitti_metrics/data_tracking_image_2/"
      "training/image_02/0000/000000.png");
  while (true) {
    cv::Mat frame;
    auto res = vc.read(frame);
    if (!res)
      break;
    auto tracks = oc_sort.update({}, frame);
  }
  return 0;
}