#include "glib-object.h"
#include "gst/gstbin.h"
#include "gst/gstbuffer.h"
#include "gst/gstelement.h"
#include "gst/gstevent.h"
#include "gst/gstmemory.h"
#include "gst/gstmeta.h"
#include "gst/gstobject.h"
#include "gst/gstpad.h"
#include "gst/gstparse.h"
#include <cstdio>
#include <glib.h>
#include <gst/gst.h>
#include <gst/rtp/gstrtpbuffer.h>
#include <gst/video/video.h>

#include <cxxopts.hpp>
#include <memory>
#include <opencv2/core/hal/interface.h>
#include <opencv2/core/types.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/opencv.hpp>
#include <string>
#include <sys/types.h>

#include "gst/gststructure.h"
#include "gst/gstvalue.h"
#include "oc_sort.hpp"
#include "utils.hpp"

using namespace std::literals::chrono_literals;

static oc_sort::OcSort tracker(oc_sort::OcSort::fromYaml("./config.yaml"));

void reflect_tracks(
    const std::vector<std::shared_ptr<oc_sort::KalmanBoxTracker>> tracks,
    const cv::Mat &mat) {
  for (int i = 0; i < tracks.size(); ++i) {
    auto detection = tracks[i];
    auto float_box = detection->get_last_observation();
    cv::Rect box = cv::Rect{
        static_cast<int>(float_box[0]), static_cast<int>(float_box[1]),
        static_cast<int>(float_box[2]), static_cast<int>(float_box[3])};
    cv::Scalar color = cv::Scalar(255, 0, 0);

    // Detection box
    cv::rectangle(mat, box, color, 1);

    // Detection box text
    std::string classString =
        std::to_string(detection->get_cls()) + ' ' +
        std::to_string(detection->get_conf()).substr(0, 4);
    cv::Size textSize =
        cv::getTextSize(classString, cv::FONT_HERSHEY_DUPLEX, 1, 2, 0);
    cv::Rect textBox(box.x, box.y - 40, textSize.width + 10,
                     textSize.height + 20);

    cv::rectangle(mat, textBox, color, cv::FILLED);
    cv::putText(mat, classString, cv::Point(box.x + 5, box.y - 10),
                cv::FONT_HERSHEY_DUPLEX, 1, cv::Scalar(0, 0, 0), 2, 0);
  }
}

GstPadProbeReturn auto_sink_probe_callback(GstPad *pad, GstPadProbeInfo *info,
                                           gpointer user_data) {
  // get result data
  if (info->type & GST_PAD_PROBE_TYPE_BUFFER) {
    GstBuffer *buffer = gst_pad_probe_info_get_buffer(info);

    GstCustomMeta *meta = gst_buffer_get_custom_meta(buffer, "tracking_meta");
    if (meta) {
      GstStructure *s = gst_custom_meta_get_structure(meta);

      GValueArray *tracks;
      gst_structure_get_array(s, "tracks", &tracks);

      for (guint i = 0; i < tracks->n_values; i++) {
        GValue *val = g_value_array_get_nth(tracks, i);
        const GstStructure *track_struct = gst_value_get_structure(val);

        gint track_id, class_id;
        gdouble confidence;
        gint x, y, w, h;

        gst_structure_get_int(track_struct, "id", &track_id);
        gst_structure_get_int(track_struct, "class_id", &class_id);
        gst_structure_get_double(track_struct, "confidence", &confidence);
        gst_structure_get_int(track_struct, "x", &x);
        gst_structure_get_int(track_struct, "y", &y);
        gst_structure_get_int(track_struct, "width", &w);
        gst_structure_get_int(track_struct, "height", &h);

        g_print("Track %d: class_id=%d, conf=%.2f, box=%d,%d,%d,%d\n", track_id,
                class_id, confidence, x, y, w, h);
      }
      g_value_array_free(tracks);
    }
  }
  return GST_PAD_PROBE_OK;
}

GstPadProbeReturn probe_callback(GstPad *pad, GstPadProbeInfo *info,
                                 gpointer user_data) {
  if (info->type & GST_PAD_PROBE_TYPE_BUFFER) {
    GstBuffer *buffer = gst_pad_probe_info_get_buffer(info);
    GstMapInfo map;

    if (!gst_buffer_map(buffer, &map, GST_MAP_READWRITE))
      return GST_PAD_PROBE_OK;

    GstCaps *caps = gst_pad_get_current_caps(pad);
    if (caps) {
      gchar *caps_str = gst_caps_to_string(caps);
      g_free(caps_str);
      gst_caps_unref(caps);
    }

    int height = tracker.params().mode_params.height_res;
    int width = tracker.params().mode_params.width_res;
    int src_type = (tracker.params().mode_params.jetson) ? CV_8UC4 : CV_8UC3;

    cv::Mat src_view(height, width, src_type, (void *)map.data,
                     map.size / height);
    cv::Mat src(height, width, CV_8UC3);

    if (src_view.type() == CV_8UC4) {
      cv::cvtColor(src_view, src, cv::COLOR_RGBA2BGR);
    } else if (src.type() == CV_8UC3) {
      src = std::move(src_view);
    }

    auto tracks = std::vector<std::shared_ptr<oc_sort::KalmanBoxTracker>>();

    tracks = tracker.update(src);
    reflect_tracks(tracks, src);

    cv::Mat out_image;
    if (tracker.params().mode_params.jetson) {
      if (src.type() == CV_8UC3) {
        cv::cvtColor(src, out_image, cv::COLOR_BGR2RGBA);
      } else if (src.type() == CV_8UC4) {
        out_image = src;
      }
    } else {
      out_image = src;
    }

    GstCustomMeta *meta = gst_buffer_add_custom_meta(buffer, "tracking_meta");
    GstStructure *main_struct = gst_custom_meta_get_structure(meta);
    GValueArray *tracks_array = g_value_array_new(tracks.size());
    for (const auto &track : tracks) {
      cv::Rect rect = cv::Rect(
          track->get_last_observation()[0], track->get_last_observation()[1],
          track->get_last_observation()[2], track->get_last_observation()[3]);
      GstStructure *track_struct = gst_structure_new(
          "track", "id", G_TYPE_INT, track->get_id(), "confidence",
          G_TYPE_DOUBLE, track->get_conf(), "class_id", G_TYPE_INT,
          track->get_cls(), "x", G_TYPE_INT, static_cast<int>(rect.x), "y",
          G_TYPE_INT, static_cast<int>(rect.y), "width", G_TYPE_INT,
          static_cast<int>(rect.width), "height", G_TYPE_INT,
          static_cast<int>(rect.height), NULL);

      GValue val = G_VALUE_INIT;
      g_value_init(&val, GST_TYPE_STRUCTURE);
      gst_value_set_structure(&val, track_struct);
      g_value_array_append(tracks_array, &val);
      gst_structure_free(track_struct);
      g_value_unset(&val);
    }
    gst_structure_set_array(main_struct, "tracks", tracks_array);
    g_value_array_free(tracks_array);

    size_t copy_size = std::min(
        (size_t)map.size, (size_t)(out_image.total() * out_image.elemSize()));

    memcpy(map.data, out_image.data, copy_size);
    gst_buffer_unmap(buffer, &map);
  }

  if (info->type & GST_PAD_PROBE_TYPE_EVENT_DOWNSTREAM) {
    GstEvent *event = GST_PAD_PROBE_INFO_EVENT(info);
    if (GST_EVENT_TYPE(event) == GST_EVENT_EOS) {
      g_main_loop_quit((GMainLoop *)user_data);
    }
  }
  return GST_PAD_PROBE_OK;
}

int main(int argc, char *argv[]) {

  GMainLoop *loop;

  GstElement *pipeline_input, *sink, *auto_sink;

  gst_init(&argc, &argv);

  loop = g_main_loop_new(NULL, FALSE);

  std::string pipeline_str;
  char pipeline_buf[2048];
  // mode specific setup
  if (tracker.params().mode_params.jetson) {
    pipeline_str =
        "rtspsrc location=%s name=mysrc !"
        "rtph264depay ! "
        "h264parse ! "
        "nvv4l2decoder ! "
        "nvvidconv ! videorate ! "
        "video/x-raw,width=%d,height=%d,framerate=%d/1,format=RGBA ! "
        "identity name=mysink signal-handoffs=true !"
        "nvvidconv ! "
        "autovideosink name=auto_sink";
  } else {
    pipeline_str =
        "rtspsrc location=%s name=mysrc !"
        "rtph264depay ! h264parse ! avdec_h264 ! "
        "videoconvert ! videorate !"
        "capsfilter "
        "caps=\"video/x-raw,width=%d,height=%d,framerate=%d/1,format=BGR\" !"
        "identity name=mysink signal-handoffs=true !"
        "videoconvert !"
        "autovideosink name=auto_sink";
  }

  snprintf(pipeline_buf, sizeof(pipeline_buf), pipeline_str.c_str(),
           tracker.params().network_params.rtsp_src.c_str(),
           tracker.params().mode_params.width_res,
           tracker.params().mode_params.height_res,
           tracker.params().mode_params.framerate);
  pipeline_input = gst_parse_launch(pipeline_buf, NULL);

  auto_sink = gst_bin_get_by_name((GstBin *)pipeline_input, "auto_sink");
  GstPad *auto_sink_pad = gst_element_get_static_pad(auto_sink, "sink");
  gst_pad_add_probe(auto_sink_pad, GST_PAD_PROBE_TYPE_DATA_DOWNSTREAM,
                    auto_sink_probe_callback, loop, NULL);

  sink = gst_bin_get_by_name((GstBin *)pipeline_input, "mysink");

  GstPad *sink_pad = gst_element_get_static_pad(sink, "src");
  gst_pad_add_probe(sink_pad, GST_PAD_PROBE_TYPE_DATA_DOWNSTREAM,
                    probe_callback, loop, NULL);

  GstBus *bus = gst_element_get_bus(pipeline_input);
  gst_bus_add_watch(
      bus,
      [](GstBus *bus, GstMessage *msg, gpointer data) -> gboolean {
        GMainLoop *loop = (GMainLoop *)data;
        switch (GST_MESSAGE_TYPE(msg)) {
        case GST_MESSAGE_ERROR: {
          GError *err = NULL;
          gchar *debug = NULL;
          gst_message_parse_error(msg, &err, &debug);
          g_printerr("Error: %s\n", err->message);
          g_error_free(err);
          g_free(debug);
          g_main_loop_quit(loop);
          break;
        }
        case GST_MESSAGE_EOS:
          g_print("End of stream\n");
          g_main_loop_quit(loop);
          break;
        default:
          break;
        }
        return TRUE;
      },
      loop);
  gst_object_unref(bus);

  static const gchar *tags = {NULL};
  gst_meta_register_custom("tracking_meta", &tags, NULL, NULL, NULL);

  g_print("Now playing: %s\n",
          tracker.params().network_params.rtsp_src.c_str());
  gst_element_set_state(pipeline_input, GST_STATE_PLAYING);

  g_print("Running...\n");
  g_main_loop_run(loop);

  g_print("Returned, stopping playback\n");
  gst_element_set_state(pipeline_input, GST_STATE_NULL);

  gst_object_unref(GST_OBJECT(pipeline_input));
  g_main_loop_unref(loop);
  gst_object_unref(pipeline_input);
  gst_object_unref(sink);
  gst_object_unref(auto_sink);

  return 0;
}
