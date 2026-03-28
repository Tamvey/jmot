#include <filesystem>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

#include "kalman_box_tracker.hpp"
#include "kalman_filter.hpp"
#include "oc_sort.hpp"

std::string coco_to_kitti_mapping(int class_id) {
  switch (class_id) {
  case 0:
    return "Pedestrian";
  case 1:
    return "Cyclist";
  case 3:
    return "Cyclist";
  case 2:
    return "Car";
  case 5:
    return "Van";
  case 7:
    return "Truck";
  case 6:
    return "Tram";
  }
  return "Misc";
}

int main() {
  std::filesystem::path videos(
      "/media/matvey/EB6B-E36F/diploma/kitti_metrics/data_tracking_image_2/"
      "training/image_02/");
  std::filesystem::create_directory(filesystem::current_path() / "results");
  std::filesystem::path results(filesystem::current_path() / "results");

  for (const auto &path : std::filesystem::directory_iterator(videos)) {
    oc_sort::KalmanBoxTracker::count = 0;
    oc_sort::OcSort::Params params = oc_sort::OcSort::fromYaml("./config.yaml");

    auto oc_sort = oc_sort::OcSort(params);

    if (path.is_directory()) {
      std::ofstream out((results / path.path().filename()).string() + ".txt");
      cv::VideoCapture vc(path.path().root_directory() /
                          path.path().relative_path() / "%06d.png");
      while (true) {
        cv::Mat frame;
        auto res = vc.read(frame);
        if (!res)
          break;
        auto tracks = oc_sort.update(frame);
        for (auto &track : tracks) {
          out << oc_sort.frame_count() << " " << track->get_id() << " "
              << coco_to_kitti_mapping(track->get_cls()) << " " << -1 << " "
              << -1 << " " << -1 << " " << track->get_last_observation()[0]
              << " " << track->get_last_observation()[1] << " "
              << track->get_last_observation()[0] +
                     track->get_last_observation()[2]
              << " "
              << track->get_last_observation()[1] +
                     track->get_last_observation()[3]
              << " " << -1 << " " << -1 << " " << -1 << " " << -1 << " " << -1
              << " " << -1 << " " << -1 << "\n";
          // auto color = cv::Scalar(255, 0, 0);
          // cv::rectangle(frame,
          //               cv::Rect(track->get_last_observation()[0],
          //                        track->get_last_observation()[1],
          //                        track->get_last_observation()[2],
          //                        track->get_last_observation()[3]),
          //               color);
          // std::string classString =
          //     std::to_string(track->get_cls()) + ' ' +
          //     std::to_string(track->get_conf()).substr(0, 4);
          // cv::Size textSize =
          //     cv::getTextSize(classString, cv::FONT_HERSHEY_DUPLEX, 1, 2, 0);
          // cv::Rect textBox(track->get_last_observation()[0],
          //                  track->get_last_observation()[1] - 40,
          //                  textSize.width + 10, textSize.height + 20);

          // cv::rectangle(frame, textBox, color, cv::FILLED);
          // cv::putText(frame, classString,
          //             cv::Point(track->get_last_observation()[0] + 5,
          //                       track->get_last_observation()[1] - 10),
          //             cv::FONT_HERSHEY_DUPLEX, 1, cv::Scalar(0, 0, 0), 2, 0);
        }
        // float scale = 1;
        // cv::resize(frame, frame,
        //            cv::Size(frame.cols * scale, frame.rows * scale));
        // cv::imshow("img", frame);
        // cv::waitKey(1);
      }
      out.flush();
    }
  }
  return 0;
}