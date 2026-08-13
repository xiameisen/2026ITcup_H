# K230D / CanMV AP-mode RTSP wireless video test for VLC
# The board creates its own WiFi hotspot. No external router is required.
#
# 1. Run this script on K230D.
# 2. Connect the receiver to AP_SSID.
# 3. Open VLC with the printed RTSP URL, usually:
#      rtsp://192.168.4.1:8554/test

from media.vencoder import *
from media.sensor import *
from media.media import *

import _thread
import multimedia as mm
import network
import os
import time
import uctypes

AP_SSID = "K230D_RTSP_AP"
AP_PASSWORD = "CHANGE_ME"
AP_START_WAIT_S = 3

RTSP_PORT = 8554
SESSION_NAME = "test"

# Keep the first AP-mode test conservative.
VIDEO_WIDTH = 800
VIDEO_HEIGHT = 480
RUN_SECONDS = 0  # 0 means run until KeyboardInterrupt / IDE stop.


def stop_sta_if_possible():
    try:
        sta = network.WLAN(network.STA_IF)
        if sta.active():
            sta.active(False)
            print("STA disabled")
    except Exception as e:
        print("STA disable skipped:", e)


def start_ap(ssid, password):
    stop_sta_if_possible()

    ap = network.WLAN(network.AP_IF)

    if ap.active():
        ap.active(False)
        time.sleep(1)

    ap.active(True)
    time.sleep(1)

    try:
        ap.config(ssid=ssid, key=password)
    except Exception as e:
        print("AP config with ssid/key failed:", e)
        ap.config(essid=ssid, password=password)

    time.sleep(AP_START_WAIT_S)

    ip, mask, gw, dns = ap.ifconfig()
    print("AP active:", ap.active())
    print("AP SSID:", ssid)
    print("AP PASSWORD:", password)
    print("AP IP:", ip)
    print("AP MASK:", mask)
    print("AP GW:", gw)
    print("AP DNS:", dns)
    print("Windows ping command: ping", ip)
    print("VLC URL: rtsp://{}:{}/{}".format(ip, RTSP_PORT, SESSION_NAME))
    return ap, ip


class RtspCameraServer:
    def __init__(self, session_name="test", port=8554, width=800, height=480):
        self.session_name = session_name
        self.port = port
        self.width = ALIGN_UP(width, 16)
        self.height = height
        self.rtspserver = mm.rtsp_server()
        self.sensor = None
        self.encoder = None
        self.link = None
        self.venc_chn = VENC_CHN_ID_0
        self.media_inited = False
        self.start_stream = False
        self.thread_done = False

    def start(self):
        self._init_video()
        self.rtspserver.rtspserver_init(self.port)
        self.rtspserver.rtspserver_createsession(
            self.session_name,
            mm.multi_media_type.media_h264,
            False,
        )
        self.rtspserver.rtspserver_start()

        self._encoder_start()
        self.sensor.run()

        self.start_stream = True
        self.thread_done = False
        _thread.start_new_thread(self._stream_loop, ())

    def stop(self):
        if self.start_stream:
            self.start_stream = False
            while not self.thread_done:
                time.sleep(0.1)

        try:
            if self.sensor:
                self.sensor.stop()
        except Exception as e:
            print("sensor.stop failed:", e)

        try:
            if self.link:
                self.link.destroy()
        except Exception as e:
            print("link.destroy failed:", e)

        try:
            if self.encoder:
                self._encoder_stop()
                self._encoder_destroy()
        except Exception as e:
            print("encoder stop/destroy failed:", e)

        try:
            if self.media_inited:
                MediaManager.deinit()
                self.media_inited = False
        except Exception as e:
            print("MediaManager.deinit failed:", e)

        try:
            self.rtspserver.rtspserver_stop()
            self.rtspserver.rtspserver_deinit()
        except Exception as e:
            print("rtsp stop/deinit failed:", e)

    def get_url(self):
        return self.rtspserver.rtspserver_getrtspurl(self.session_name)

    def _init_video(self):
        self.sensor = Sensor()
        self.sensor.reset()
        self.sensor.set_framesize(width=self.width, height=self.height, alignment=12)
        self.sensor.set_pixformat(Sensor.YUV420SP)

        self.encoder = Encoder()
        self._encoder_set_out_bufs()
        attr = ChnAttrStr(
            self.encoder.PAYLOAD_TYPE_H264,
            self.encoder.H264_PROFILE_MAIN,
            self.width,
            self.height,
        )
        self._encoder_create(attr)

        self.link = MediaManager.link(
            self.sensor.bind_info()["src"],
            (VIDEO_ENCODE_MOD_ID, VENC_DEV_ID, self.venc_chn),
        )
        MediaManager.init()
        self.media_inited = True

    def _encoder_set_out_bufs(self):
        try:
            self.encoder.SetOutBufs(self.venc_chn, 8, self.width, self.height)
        except TypeError:
            self.encoder.SetOutBufs(8, self.width, self.height)

    def _encoder_create(self, attr):
        try:
            self.encoder.Create(self.venc_chn, attr)
        except TypeError:
            self.encoder.Create(attr)
            try:
                self.venc_chn = self.encoder.chn
            except Exception:
                pass

    def _encoder_start(self):
        try:
            self.encoder.Start(self.venc_chn)
        except TypeError:
            self.encoder.Start()

    def _encoder_get_stream(self, stream_data):
        try:
            return self.encoder.GetStream(stream_data)
        except TypeError:
            return self.encoder.GetStream(self.venc_chn, stream_data)

    def _encoder_release_stream(self, stream_data):
        try:
            return self.encoder.ReleaseStream(stream_data)
        except TypeError:
            return self.encoder.ReleaseStream(self.venc_chn, stream_data)

    def _encoder_stop(self):
        try:
            return self.encoder.Stop(self.venc_chn)
        except TypeError:
            return self.encoder.Stop()

    def _encoder_destroy(self):
        try:
            return self.encoder.Destroy(self.venc_chn)
        except TypeError:
            return self.encoder.Destroy()

    def _stream_loop(self):
        stream_data = StreamData()
        frame_count = 0
        try:
            while self.start_stream:
                os.exitpoint()
                ret = self._encoder_get_stream(stream_data)
                if ret != 0:
                    time.sleep_ms(5)
                    continue

                for pack_idx in range(stream_data.pack_cnt):
                    data_size = stream_data.data_size[pack_idx]
                    data = bytes(uctypes.bytearray_at(stream_data.data[pack_idx], data_size))
                    try:
                        timestamp = stream_data.pts[pack_idx]
                    except Exception:
                        timestamp = time.ticks_ms()

                    self.rtspserver.rtspserver_sendvideodata(
                        self.session_name,
                        data,
                        data_size,
                        timestamp,
                    )

                self._encoder_release_stream(stream_data)
                frame_count += 1
                if frame_count % 30 == 0:
                    print("RTSP sent frames:", frame_count)

        except BaseException as e:
            import sys
            sys.print_exception(e)
        finally:
            self.thread_done = True


if __name__ == "__main__":
    os.exitpoint(os.EXITPOINT_ENABLE)

    ap, ap_ip = start_ap(AP_SSID, AP_PASSWORD)

    server = RtspCameraServer(
        session_name=SESSION_NAME,
        port=RTSP_PORT,
        width=VIDEO_WIDTH,
        height=VIDEO_HEIGHT,
    )

    try:
        server.start()
        print("RTSP server started:", server.get_url())
        print("Use VLC URL: rtsp://{}:{}/{}".format(ap_ip, RTSP_PORT, SESSION_NAME))

        start = time.time()
        while True:
            if RUN_SECONDS > 0 and time.time() - start >= RUN_SECONDS:
                break
            time.sleep(1)

    except KeyboardInterrupt:
        print("User stopped")
    finally:
        server.stop()
        print("done")
