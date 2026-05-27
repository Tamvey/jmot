#include "glib.h"
#include <gst/gst.h>

#include <gst/rtsp-server/rtsp-server.h>
#include <string>

#define DEFAULT_RTSP_PORT "8554"

static char *port = (char *)DEFAULT_RTSP_PORT;

static GOptionEntry entries[] = {
    {"port", 'p', 0, G_OPTION_ARG_STRING_ARRAY, &port,
     "Port to listen on (default: " DEFAULT_RTSP_PORT ")", "PORT"},
    {NULL}};

int main(int argc, char *argv[]) {
  GMainLoop *loop;
  GstRTSPServer *server;
  GstRTSPMountPoints *mounts;
  GstRTSPMediaFactory *factory;
  GOptionContext *optctx;
  GError *error = NULL;
  gchar *str, *str_path;

  optctx = g_option_context_new("<filename.mp4> - Test RTSP Server, MP4");
  g_option_context_add_main_entries(optctx, entries, NULL);
  g_option_context_add_group(optctx, gst_init_get_option_group());
  if (!g_option_context_parse(optctx, &argc, &argv, &error)) {
    g_printerr("Error parsing options: %s\n", error->message);
    g_option_context_free(optctx);
    g_clear_error(&error);
    return -1;
  }

  g_option_context_free(optctx);

  loop = g_main_loop_new(NULL, FALSE);

  server = gst_rtsp_server_new();
  g_object_set(server, "service", port, NULL);

  mounts = gst_rtsp_server_get_mount_points(server);

  for (int i = 1; i < argc; i++) {
    std::string path{argv[i]};
    if (path.find("/dev/") == path.npos) {
      str = g_strdup_printf("( "
                            "filesrc location=\"%s\" ! qtdemux name=d "
                            "d. ! queue ! rtph264pay pt=96 name=pay0 "
                            "d. ! queue ! rtpmp4apay pt=97 name=pay1 "
                            ")",
                            path.c_str());
    } else {
      str = g_strdup_printf(""
                            "v4l2src device=\"%s\" ! nvvidconv ! "
                            "nvv4l2h264enc ! h264parse ! rtph264pay name=pay0"
                            "",
                            path.c_str());
    }

    str_path = g_strdup_printf("/test_%d", i);
    factory = gst_rtsp_media_factory_new();
    gst_rtsp_media_factory_set_launch(factory, str);

    gst_rtsp_mount_points_add_factory(mounts, str_path, factory);
    g_free(str);
    g_free(str_path);
  }
  g_object_unref(mounts);

  gst_rtsp_server_attach(server, NULL);

  g_print("stream ready at rtsp://127.0.0.1:%s\n", port);
  g_main_loop_run(loop);

  return 0;
}
