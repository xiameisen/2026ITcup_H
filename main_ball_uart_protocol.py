import gc
import time

from libs.PipeLine import PipeLine
from libs.YOLO import YOLO11
from media.sensor import Sensor
from ybUtils.YbUart import YbUart


KMODEL_PATH = "/sdcard/yolo11n_det_320.kmodel"
LABELS = {0: "ball"}
MODEL_INPUT_SIZE = [320, 320]

DISPLAY = "lcd2_4"

if DISPLAY == "hdmi":
    DISPLAY_MODE = "hdmi"
    DISPLAY_SIZE = [1920, 1080]
    SENSOR_SIZE = [1920, 1080]
    RGB888P_SIZE = [640, 360]
elif DISPLAY == "lcd3_5":
    DISPLAY_MODE = "st7701"
    DISPLAY_SIZE = [800, 480]
    SENSOR_SIZE = [1920, 1080]
    RGB888P_SIZE = [640, 360]
else:
    DISPLAY_MODE = "st7701"
    DISPLAY_SIZE = [640, 480]
    SENSOR_SIZE = [1280, 960]
    RGB888P_SIZE = [640, 480]

CONFIDENCE_THRESHOLD = 0.60
NMS_THRESHOLD = 0.45

UART_BAUDRATE = 115200
UART_SEND_INTERVAL_MS = 50

# Calibrate this value after measuring the real scene.
# Default: full image width maps to 100 mm, center is 0 mm.
VIEW_WIDTH_MM = 100

FPS_X = 8
FPS_Y = 8
TEXT_X = 8
TEXT_Y = 42
FONT_SIZE = 24
TEXT_COLOR = (255, 255, 0, 255)


def ticks_ms():
    try:
        return time.ticks_ms()
    except AttributeError:
        return int(time.time() * 1000)


def ticks_diff(now, old):
    try:
        return time.ticks_diff(now, old)
    except AttributeError:
        return now - old


def clamp(value, low, high):
    if value < low:
        return low
    if value > high:
        return high
    return value


def draw_text(osd_img, x, y, text):
    try:
        osd_img.draw_string_advanced(x, y, FONT_SIZE, text, color=TEXT_COLOR)
    except AttributeError:
        osd_img.draw_string(x, y, text, color=TEXT_COLOR, scale=2)


def is_valid_box(x1, y1, x2, y2):
    return x2 > x1 and y2 > y1


def parse_detection(det):
    if len(det) < 6:
        return None

    # Format A: [class_id, score, x1, y1, x2, y2]
    cls_a = int(det[0])
    score_a = float(det[1])
    x1_a = float(det[2])
    y1_a = float(det[3])
    x2_a = float(det[4])
    y2_a = float(det[5])
    if cls_a in LABELS and is_valid_box(x1_a, y1_a, x2_a, y2_a):
        return cls_a, score_a, x1_a, y1_a, x2_a, y2_a

    # Format B: [x1, y1, x2, y2, score, class_id]
    x1_b = float(det[0])
    y1_b = float(det[1])
    x2_b = float(det[2])
    y2_b = float(det[3])
    score_b = float(det[4])
    cls_b = int(det[5])
    if cls_b in LABELS and is_valid_box(x1_b, y1_b, x2_b, y2_b):
        return cls_b, score_b, x1_b, y1_b, x2_b, y2_b

    return None


def normalize_score(score):
    if score > 1.0:
        score = score / 100.0
    return score


def choose_best_ball(dets):
    if dets is None or len(dets) == 0:
        return None

    best_info = None
    best_score = -1.0
    for det in dets:
        info = parse_detection(det)
        if info is None:
            continue
        cls_id, score, x1, y1, x2, y2 = info
        if cls_id != 0:
            continue
        score = normalize_score(score)
        if score > best_score:
            best_score = score
            best_info = info
    return best_info


def ball_to_x_mm(ball_info):
    cls_id, score, x1, y1, x2, y2 = ball_info
    center_x = (x1 + x2) * 0.5
    image_center_x = RGB888P_SIZE[0] * 0.5
    x_mm = (center_x - image_center_x) * VIEW_WIDTH_MM / RGB888P_SIZE[0]
    return int(round(x_mm))


def make_packet(x_mm, v_mm_s, confidence, state):
    x_mm = int(clamp(x_mm, -999, 999))
    v_mm_s = int(clamp(v_mm_s, -999, 999))
    confidence = int(clamp(confidence, 0, 100))
    state = int(clamp(state, 0, 1))
    return "$B,%+04d,%+04d,%03d,%d\n" % (x_mm, v_mm_s, confidence, state)


def main():
    print("K230 YOLO11 ball UART protocol start")
    print("uart:", UART_BAUDRATE)
    print("format: $B,%+04d,%+04d,%03d,%d\\n")

    uart = YbUart(baudrate=UART_BAUDRATE)

    pl = PipeLine(
        rgb888p_size=RGB888P_SIZE,
        display_size=DISPLAY_SIZE,
        display_mode=DISPLAY_MODE,
    )
    pl.create(sensor=Sensor(id=2, width=SENSOR_SIZE[0], height=SENSOR_SIZE[1]))
    display_size = pl.get_display_size()

    yolo = YOLO11(
        task_type="detect",
        mode="video",
        kmodel_path=KMODEL_PATH,
        labels=LABELS,
        rgb888p_size=RGB888P_SIZE,
        model_input_size=MODEL_INPUT_SIZE,
        display_size=display_size,
        conf_thresh=CONFIDENCE_THRESHOLD,
        nms_thresh=NMS_THRESHOLD,
        max_boxes_num=50,
        debug_mode=0,
    )
    yolo.config_preprocess()

    clock = time.clock()
    last_x_mm = 0
    last_detect_ms = ticks_ms()
    last_send_ms = 0

    try:
        while True:
            clock.tick()
            now_ms = ticks_ms()

            img = pl.get_frame()
            res = yolo.run(img)
            yolo.draw_result(res, pl.osd_img)

            ball_info = choose_best_ball(res)
            if ball_info is not None:
                cls_id, score, x1, y1, x2, y2 = ball_info
                x_mm = ball_to_x_mm(ball_info)
                dt_ms = ticks_diff(now_ms, last_detect_ms)
                if dt_ms > 0:
                    v_mm_s = int(round((x_mm - last_x_mm) * 1000 / dt_ms))
                else:
                    v_mm_s = 0
                confidence = int(round(normalize_score(score) * 100))
                state = 1
                last_x_mm = x_mm
                last_detect_ms = now_ms
            else:
                x_mm = 0
                v_mm_s = 0
                confidence = 0
                state = 0

            packet = make_packet(x_mm, v_mm_s, confidence, state)

            if ticks_diff(now_ms, last_send_ms) >= UART_SEND_INTERVAL_MS:
                uart.send(packet)
                last_send_ms = now_ms

            fps = clock.fps()
            draw_text(pl.osd_img, FPS_X, FPS_Y, "FPS: %.1f" % fps)
            draw_text(pl.osd_img, TEXT_X, TEXT_Y, packet.strip())

            print(packet)
            pl.show_image()
            gc.collect()
    except Exception as e:
        print("stopped:", e)
    finally:
        try:
            yolo.deinit()
        except Exception:
            pass
        try:
            pl.destroy()
        except Exception:
            pass
        try:
            uart.deinit()
        except Exception:
            pass
        gc.collect()


if __name__ == "__main__":
    main()
