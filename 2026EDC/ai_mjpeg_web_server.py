# K230 CanMV steel-ball AI MJPEG web server
#
# Features:
#   1. YOLOv8 steel-ball detection instead of face detection.
#   2. The browser view is rotated 90 degrees counter-clockwise.
#   3. The browser can record the displayed view and download a WebM file.
#   4. The IP address is shown at the top-left of the K230 LCD OSD.

import gc
import network
import os
import _thread
import socket
import sys
import time

from libs.PipeLine import PipeLine
from libs.YOLO import YOLOv8
from media.display import Display
from media.media import *
from media.mjpeg import MJPEGEncoder


# -------------------- User settings --------------------
WIFI_SSID = "NX30PRO_2.4G"
WIFI_PASSWORD = "qwe12345"

SERVER_PORT = 8080
RGB888P_SIZE = [640, 480]
DISPLAY_SIZE = [640, 480]
DISPLAY_MODE = "lcd"

KMODEL_PATH = "/sdcard/model.kmodel"
LABELS = ["ball"]
MODEL_INPUT_SIZE = [640, 640]
CONF_THRESH = 0.5
NMS_THRESH = 0.5
MAX_BOXES = 50

STREAM_FPS = 10
JPEG_QUALITY = 40
# Some RT-Smart sockets report a temporary no-data state as errno 110 while
# a browser is opening a connection. Keep the request parser alive longer.
REQUEST_TIMEOUT_MS = 1000
SEND_STALL_TIMEOUT_MS = 300
SEND_CHUNK_BYTES = 4096
MAX_REQUEST_BYTES = 4096

ai_stop = False


# The page rotates the complete composed image. LCD remains in its established
# orientation, while the web view and its downloaded recording are CCW 90 deg.
INDEX_HTML = b"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>K230 Steel Ball Camera</title>
<style>
*{box-sizing:border-box}
html,body{width:100%;height:100%;margin:0;background:#080808;color:#f5f5f5;
font-family:Arial,sans-serif;overflow:hidden}
body{display:grid;grid-template-rows:48px 1fr 50px}
header{display:flex;align-items:center;justify-content:space-between;padding:0 14px;
background:#151515;border-bottom:1px solid #2b2b2b;font-size:14px}
#status{color:#55dd77}
main{min-width:0;min-height:0;display:flex;align-items:center;justify-content:center;
position:relative;overflow:hidden}
#viewer{width:min(90vw,70vh);height:min(90vh,120vw);aspect-ratio:3/4;
position:relative;display:flex;align-items:center;justify-content:center;background:#000}
#stream{display:block;width:100%;height:100%;object-fit:contain;transform:rotate(-90deg)}
#ip{position:absolute;left:8px;top:8px;color:#55ff77;background:rgba(0,0,0,.65);
padding:3px 6px;border-radius:3px;font-size:13px;z-index:3}
#hint{position:absolute;bottom:8px;left:0;right:0;text-align:center;color:#999;
font-size:11px;z-index:3}
footer{display:flex;align-items:center;justify-content:center;gap:10px;
background:#151515;border-top:1px solid #2b2b2b}
button{border:1px solid #4772aa;border-radius:4px;background:#19314f;color:#fff;
padding:6px 12px;font-size:13px}
button:disabled{opacity:.5}
#recordStatus{font-size:12px;color:#ffcc66;min-width:90px}
</style>
</head>
<body>
<header><span>K230 Steel Ball AI Camera</span><span id="status">connecting</span></header>
<main>
  <div id="viewer">
    <img id="stream" alt="steel ball stream">
    <div id="hint">The web view is rotated CCW 90 degrees</div>
  </div>
</main>
<footer>
  <button id="recordButton">Start recording</button>
  <button id="downloadButton" disabled>Download video</button>
  <span id="recordStatus">not recording</span>
</footer>
<canvas id="recordCanvas" style="display:none"></canvas>
<script>
const stream=document.getElementById("stream");
const status=document.getElementById("status");
const recordButton=document.getElementById("recordButton");
const downloadButton=document.getElementById("downloadButton");
const recordStatus=document.getElementById("recordStatus");
const canvas=document.getElementById("recordCanvas");
const ctx=canvas.getContext("2d");
let streamUrl=null;
let activeController=null;
let connectionStartedAt=0;
let lastFrameAt=0;
let recorder=null;
let recordedChunks=[];
let recordedBlob=null;
let drawTimer=null;
let recordStarted=0;

function sleep(ms){return new Promise(resolve=>setTimeout(resolve,ms))}
function markerIndex(data,a,b,start){
  for(let i=start;i+1<data.length;i++){
    if(data[i]===a&&data[i+1]===b)return i;
  }
  return -1;
}
function appendBytes(left,right){
  const merged=new Uint8Array(left.length+right.length);
  merged.set(left); merged.set(right,left.length); return merged;
}
function showJpeg(bytes){
  const url=URL.createObjectURL(new Blob([bytes],{type:"image/jpeg"}));
  const old=streamUrl; streamUrl=url; stream.src=url;
  if(old)URL.revokeObjectURL(old);
  lastFrameAt=Date.now();
  status.textContent="online";
}
async function connectStream(){
  while(true){
    const controller=new AbortController();
    activeController=controller;
    connectionStartedAt=Date.now();
    lastFrameAt=0;
    try{
      status.textContent="connecting";
      const response=await fetch("/stream?_="+Date.now(),{cache:"no-store",signal:controller.signal});
      if(!response.ok||!response.body)throw new Error("stream unavailable");
      const reader=response.body.getReader();
      let buffer=new Uint8Array(0);
      while(true){
        const packet=await reader.read();
        if(packet.done)throw new Error("stream closed");
        buffer=appendBytes(buffer,packet.value);
        while(true){
          const start=markerIndex(buffer,0xff,0xd8,0);
          if(start<0){
            buffer=buffer.slice(Math.max(0,buffer.length-1)); break;
          }
          const end=markerIndex(buffer,0xff,0xd9,start+2);
          if(end<0){buffer=buffer.slice(start); break}
          showJpeg(buffer.slice(start,end+2));
          buffer=buffer.slice(end+2);
        }
      }
    }catch(error){
      status.textContent="reconnecting";
      if(activeController===controller)activeController=null;
      if(streamUrl){URL.revokeObjectURL(streamUrl);streamUrl=null}
      stream.removeAttribute("src");
      await sleep(500);
    }
  }
}

// Some browsers keep a failed fetch reader pending after a TCP half-close.
// Abort it explicitly when no JPEG has arrived for two seconds.
setInterval(()=>{
  if(!activeController)return;
  const reference=lastFrameAt||connectionStartedAt;
  if(Date.now()-reference>2000)activeController.abort();
},500);

function drawRotatedFrame(){
  if(!stream.complete || !stream.naturalWidth) return;
  const w=stream.naturalWidth;
  const h=stream.naturalHeight;
  if(canvas.width!==h || canvas.height!==w){canvas.width=h;canvas.height=w}
  ctx.fillStyle="#000";
  ctx.fillRect(0,0,canvas.width,canvas.height);
  ctx.save();
  ctx.translate(0,canvas.height);
  ctx.rotate(-Math.PI/2);
  ctx.drawImage(stream,0,0,w,h);
  ctx.restore();
}

function supportedMime(){
  // Prefer MP4/H.264 for Windows playback. Fall back only when the browser
  // does not expose an MP4 MediaRecorder encoder.
  const types=[
    "video/mp4;codecs=avc1.42E01E",
    "video/mp4",
    "video/webm;codecs=vp9",
    "video/webm;codecs=vp8",
    "video/webm"
  ];
  for(const type of types){
    if(MediaRecorder.isTypeSupported(type)) return type;
  }
  return "";
}

function startRecording(){
  if(!stream.complete || !stream.naturalWidth){
    recordStatus.textContent="waiting for video";
    return;
  }
  drawRotatedFrame();
  const fps=10;
  const mediaStream=canvas.captureStream(fps);
  const mime=supportedMime();
  const isMp4=mime.indexOf("video/mp4")===0;
  try{recorder=mime?new MediaRecorder(mediaStream,{mimeType:mime}):new MediaRecorder(mediaStream)}
  catch(error){recordStatus.textContent="recording unsupported";return}
  recordedChunks=[];
  recordedBlob=null;
  downloadButton.disabled=true;
  recorder.ondataavailable=(event)=>{if(event.data&&event.data.size)recordedChunks.push(event.data)};
  recorder.onstop=()=>{
    recordedBlob=new Blob(recordedChunks,{type:recorder.mimeType||"video/webm"});
    downloadButton.disabled=false;
    recordStatus.textContent=(recordedBlob.type.indexOf("mp4")>=0?"MP4":"WebM")+" ready";
    if(drawTimer){clearInterval(drawTimer);drawTimer=null}
    mediaStream.getTracks().forEach(track=>track.stop());
  };
  // Avoid timesliced MP4 fragments. A single finalized MP4 is more likely to
  // contain a complete duration/index and therefore supports seeking on
  // Windows players. WebM can still use short chunks safely.
  if(isMp4) recorder.start();
  else recorder.start(200);
  drawTimer=setInterval(drawRotatedFrame,1000/fps);
  recordStarted=Date.now();
  recordButton.textContent="Stop recording";
  recordStatus.textContent="recording 00:00";
}

function stopRecording(){
  if(!recorder||recorder.state!=="recording")return;
  recorder.stop();
  recordButton.textContent="Start recording";
}

recordButton.onclick=()=>{
  if(recorder&&recorder.state==="recording")stopRecording();
  else startRecording();
};
downloadButton.onclick=()=>{
  if(!recordedBlob)return;
  const url=URL.createObjectURL(recordedBlob);
  const link=document.createElement("a");
  link.href=url;
  const extension=recordedBlob.type.indexOf("mp4")>=0?"mp4":"webm";
  link.download="k230_steel_ball_"+Date.now()+"."+extension;
  link.click();
  setTimeout(()=>URL.revokeObjectURL(url),1000);
};
setInterval(()=>{
  if(recorder&&recorder.state==="recording"){
    const sec=Math.floor((Date.now()-recordStarted)/1000);
    recordStatus.textContent="recording "+String(Math.floor(sec/60)).padStart(2,"0")+":"+String(sec%60).padStart(2,"0");
  }
},500);
window.addEventListener("beforeunload",()=>{
  if(recorder&&recorder.state==="recording")recorder.stop();
});
connectStream();
</script>
</body>
</html>
"""


def wait_for_ip(netif, timeout_ms=20000):
    started = time.ticks_ms()
    while netif.ifconfig()[0] == "0.0.0.0":
        if time.ticks_diff(time.ticks_ms(), started) >= timeout_ms:
            raise RuntimeError("network address timeout")
        os.exitpoint()
        time.sleep_ms(100)
    return netif.ifconfig()[0]


def connect_network():
    netif = network.WLAN(network.STA_IF)
    netif.active(True)
    print("[WIFI] connecting:", WIFI_SSID)
    if not netif.isconnected():
        if not netif.connect(WIFI_SSID, WIFI_PASSWORD):
            raise RuntimeError("Wi-Fi connect failed")
        wait_for_ip(netif)
    ip = netif.ifconfig()[0]
    print("[WIFI] connected, ifconfig:", netif.ifconfig())
    return netif, ip


def draw_ip(osd_img, ip):
    # draw_string is deprecated on current CanMV firmware.
    # IP is intentionally drawn only on the K230 LCD OSD, not in HTML.
    osd_img.draw_string_advanced(
        8,
        8,
        24,
        "IP: " + ip,
        color=(0, 255, 80, 255),
    )


def capture_jpeg(pipeline, encoder, osd_lock):
    started = time.ticks_ms()

    # The camera video layer is bound directly to the display and keeps
    # refreshing independently. Only synchronize the OSD + writeback capture.
    osd_lock.acquire()
    try:
        pipeline.show_image()
        display_frame = Display.writeback_dump(1000)
    finally:
        osd_lock.release()

    if display_frame is None:
        raise RuntimeError("display writeback timeout")
    jpeg = encoder.encode(display_frame, timeout_ms=1000)
    del display_frame
    return jpeg, time.ticks_diff(time.ticks_ms(), started)


def ai_worker(pipeline, detector, ip, osd_lock):
    global ai_stop
    frame_count = 0

    while not ai_stop:
        frame = None
        try:
            frame = pipeline.get_frame()
            if frame is None:
                time.sleep_ms(5)
                continue

            result = detector.run(frame)
            # Always clear the previous OSD first. This prevents an old ball
            # box from remaining visible after detection is lost.
            osd_lock.acquire()
            try:
                pipeline.osd_img.clear()
                detector.draw_result(result, pipeline.osd_img)
                draw_ip(pipeline.osd_img, ip)
                # Commit the new OSD immediately. Without this call the
                # in-memory detection boxes change, but the LCD keeps showing
                # the first submitted OSD frame until the HTTP loop runs.
                pipeline.show_image()
            finally:
                osd_lock.release()

            frame_count += 1
            if frame_count % 10 == 0:
                gc.collect()
        except BaseException as error:
            print("[HTTP] AI worker error:", error)
            time.sleep_ms(50)
        finally:
            if frame is not None:
                del frame


def send_all(client, data):
    view = memoryview(data)
    offset = 0
    deadline = time.ticks_add(time.ticks_ms(), SEND_STALL_TIMEOUT_MS)
    while offset < len(view):
        try:
            end = min(offset + SEND_CHUNK_BYTES, len(view))
            sent = client.send(view[offset:end])
            if sent:
                offset += sent
                deadline = time.ticks_add(time.ticks_ms(), SEND_STALL_TIMEOUT_MS)
                continue
        except OSError as error:
            if error.errno not in (11, 110):
                raise
        if time.ticks_diff(deadline, time.ticks_ms()) <= 0:
            raise OSError(110)
        os.exitpoint()
        time.sleep_ms(1)


def read_path(client):
    request = bytearray()
    deadline = time.ticks_add(time.ticks_ms(), REQUEST_TIMEOUT_MS)
    client.setblocking(False)
    while len(request) < MAX_REQUEST_BYTES:
        try:
            chunk = client.recv(min(256, MAX_REQUEST_BYTES - len(request)))
        except OSError as error:
            # errno 110 with an empty request is usually a browser preconnect
            # that has not sent HTTP data. Release it immediately so it cannot
            # block the single-client server from accepting the real request.
            if error.errno == 110 and len(request) == 0:
                raise
            if error.errno != 11:
                raise
            chunk = None
        if chunk:
            request.extend(chunk)
            if request.find(b"\r\n\r\n") >= 0:
                break
        elif time.ticks_diff(deadline, time.ticks_ms()) <= 0:
            raise OSError(110)
        else:
            os.exitpoint()
            time.sleep_ms(10)
    request_line = bytes(request).split(b"\r\n", 1)[0].split()
    if len(request_line) < 2:
        return "/"
    return request_line[1].decode().split("?", 1)[0]


def send_page(client, ip):
    page = INDEX_HTML
    send_all(
        client,
        b"HTTP/1.1 200 OK\r\n"
        b"Content-Type: text/html; charset=utf-8\r\n"
        b"Cache-Control: no-store\r\n"
        b"Connection: close\r\n\r\n" + page,
    )


def stream_mjpeg(client, pipeline, encoder, osd_lock):
    send_all(
        client,
        b"HTTP/1.1 200 OK\r\n"
        b"Content-Type: multipart/x-mixed-replace; boundary=frame\r\n"
        b"Cache-Control: no-store, no-cache, must-revalidate\r\n"
        b"Pragma: no-cache\r\nConnection: close\r\n\r\n",
    )
    interval = 1000 // STREAM_FPS
    count = 0
    stat_started = time.ticks_ms()
    capture_total = 0
    send_total = 0
    print("[HTTP] steel-ball MJPEG client started")
    while True:
        loop_started = time.ticks_ms()
        jpeg, capture_ms = capture_jpeg(pipeline, encoder, osd_lock)
        jpeg_size = len(jpeg)
        part_header = (
            "--frame\r\nContent-Type: image/jpeg\r\n"
            "Content-Length: %d\r\n\r\n" % jpeg_size
        )
        send_started = time.ticks_ms()
        try:
            send_all(client, part_header.encode())
            send_all(client, jpeg)
            send_all(client, b"\r\n")
        except OSError as error:
            print("[HTTP] client dropped, send errno:", error.errno)
            del jpeg
            raise
        send_ms = time.ticks_diff(time.ticks_ms(), send_started)
        del jpeg
        count += 1
        capture_total += capture_ms
        send_total += send_ms
        if count % 10 == 0:
            elapsed = max(1, time.ticks_diff(time.ticks_ms(), stat_started))
            print(
                "[HTTP] frames=%d fps=%.1f jpeg=%dB capture=%dms send=%dms"
                % (count, count * 1000.0 / elapsed, jpeg_size,
                   capture_total // count, send_total // count)
            )
            gc.collect()
        delay = interval - time.ticks_diff(time.ticks_ms(), loop_started)
        if delay > 0:
            time.sleep_ms(delay)
        os.exitpoint()


def client_worker(client, pipeline, encoder, osd_lock, ip, stream_lock):
    """Handle one browser connection without blocking accept()."""
    try:
        path = read_path(client)
        print("[HTTP] request:", path)
        if path == "/stream":
            # Only one MJPEG client may use the hardware encoder at a time.
            stream_lock.acquire()
            try:
                stream_mjpeg(client, pipeline, encoder, osd_lock)
            finally:
                stream_lock.release()
        else:
            send_page(client, ip)
    except OSError as error:
        print("[HTTP] client ended, errno:", error.errno)
    except BaseException as error:
        print("[HTTP] client error:", error)
        sys.print_exception(error)
    finally:
        client.close()


def main():
    global ai_stop
    netif = None
    pipeline = None
    detector = None
    encoder = None
    server = None
    osd_lock = None
    stream_lock = None
    ai_thread_started = False
    writeback_enabled = False
    try:
        netif, ip = connect_network()
        pipeline = PipeLine(
            rgb888p_size=RGB888P_SIZE,
            display_size=DISPLAY_SIZE,
            display_mode=DISPLAY_MODE,
        )
        pipeline.create(to_ide=False)
        detector = YOLOv8(
            task_type="detect",
            mode="video",
            kmodel_path=KMODEL_PATH,
            labels=LABELS,
            rgb888p_size=RGB888P_SIZE,
            model_input_size=MODEL_INPUT_SIZE,
            display_size=DISPLAY_SIZE,
            conf_thresh=CONF_THRESH,
            nms_thresh=NMS_THRESH,
            max_boxes_num=MAX_BOXES,
            debug_mode=0,
        )
        detector.config_preprocess()
        encoder = MJPEGEncoder(quality=JPEG_QUALITY)
        if not Display.writeback(True):
            raise RuntimeError("start display writeback failed")
        writeback_enabled = True

        for _ in range(5):
            pipeline.get_frame()

        osd_lock = _thread.allocate_lock()
        stream_lock = _thread.allocate_lock()
        ai_stop = False
        _thread.start_new_thread(ai_worker, (pipeline, detector, ip, osd_lock))
        ai_thread_started = True

        time.sleep_ms(100)
        first_jpeg, first_ms = capture_jpeg(pipeline, encoder, osd_lock)
        print("[INIT] first steel-ball JPEG:", len(first_jpeg), "bytes,", first_ms, "ms")
        del first_jpeg
        gc.collect()

        server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        server.bind(socket.getaddrinfo("0.0.0.0", SERVER_PORT)[0][-1])
        # 浏览器打开/关闭页面时可能同时产生多个预连接。增大 backlog，
        # 避免其中一个预连接被取消后影响后续真正的 /stream 连接。
        server.listen(4)
        server.setblocking(False)
        print("[HTTP] open http://%s:%d/" % (ip, SERVER_PORT))
        print("[HTTP] web view rotation: CCW 90 degrees")

        while True:
            try:
                accepted_client, address = server.accept()
            except OSError as error:
                # 11: EAGAIN，当前没有新连接。
                # 103: ECONNABORTED，客户端在 accept 前主动关闭/取消连接。
                # 104: ECONNRESET，客户端重置连接。
                # 110: ETIMEDOUT，连接建立或等待过程中超时。
                # 这些都是浏览器断开时可能出现的正常网络事件，不能
                # 让它们冒泡到外层，否则整个 HTTP 程序会退出。
                if error.errno not in (11, 103, 104, 110):
                    raise
                os.exitpoint()
                time.sleep_ms(20)
                continue
            print("[HTTP] accepted:", address)
            _thread.start_new_thread(
                client_worker,
                (accepted_client, pipeline, encoder, osd_lock, ip, stream_lock),
            )
            gc.collect()
    except KeyboardInterrupt:
        print("[APP] stopped by user")
    except BaseException as error:
        print("[APP] fatal error:", error)
        sys.print_exception(error)
    finally:
        ai_stop = True
        if ai_thread_started:
            time.sleep_ms(100)
        if server is not None:
            server.close()
        if encoder is not None:
            encoder.close()
        if detector is not None:
            detector.deinit()
        if writeback_enabled:
            Display.writeback(False)
        if pipeline is not None:
            pipeline.destroy()
        netif = None
        os.exitpoint(os.EXITPOINT_ENABLE_SLEEP)
        time.sleep_ms(100)


if __name__ == "__main__":
    os.exitpoint(os.EXITPOINT_ENABLE)
    main()
