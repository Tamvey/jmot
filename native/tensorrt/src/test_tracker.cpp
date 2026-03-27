#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

#include "oc_sort.hpp"

int main() {
  oc_sort::OcSort::Params params = {};
  auto oc_sort = oc_sort::OcSort(params);

  // cv::VideoCapture vc(
  //     "/media/matvey/EB6B-E36F/diploma/kitti_metrics/data_tracking_image_2/"
  //     "training/image_02/0000/%06d.png");
  cv::VideoCapture vc("/home/matvey/Videos/work_videos/cut_video.mp4");
  while (true) {
    cv::Mat frame;
    auto res = vc.read(frame);
    if (!res)
      break;
    auto tracks = oc_sort.update(frame);
    for (auto &track : tracks) {
      cv::rectangle(frame,
                    cv::Rect(track->get_last_observation()[0],
                             track->get_last_observation()[1],
                             track->get_last_observation()[2],
                             track->get_last_observation()[3]),
                    cv::Scalar(255, 0, 0));
    }
    float scale = 0.7;
    cv::resize(frame, frame, cv::Size(frame.cols * scale, frame.rows * scale));
    cv::imshow("img", frame);
    cv::waitKey(20);
  }
  return 0;
}