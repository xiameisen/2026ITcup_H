import gc
import time

from libs.PipeLine import PipeLine
from libs.YOLO import YOLOv8
from media.sensor import Sensor
from ybUtils.YbUart import YbUart


# Model package from 2026EDC.
KMODEL_PATH = "/sdcard/model.kmodel"
LABELS = ["ball"]
BALL_CLASS_ID = 0
MODEL_INPUT_SIZE = [640, 640]

# Use the 4:3 path so the board-side image size stays close to 640x480.
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

CONFIDENCE_THRESHOLD = 0.50
NMS_THRESHOLD = 0.50

UART_BAUDRATE = 115200
UART_SEND_INTERVAL_MS = 50  # 50 ms = 20 Hz UART output.

# Calibration:
# Put the ball at the rod center O and set ROD_CENTER_PX to its image x coordinate.
# Put the ball 50 mm away from O and set PIXEL_PER_MM = abs(x_50mm_px - ROD_CENTER_PX) / 50.
ROD_CENTER_PX = RGB888P_SIZE[0] // 2  # Ball center pixel when the ball is at rod center O.
PIXEL_PER_MM = 6.4  # Camera scale. Must be calibrated with the ruler on the rod.

POSITION_FILTER_ALPHA = 0.30  # Larger value = faster response, smaller value = smoother data.

FONT_SIZE = 24
TEXT_COLOR = (255, 255, 0, 255)
BOX_COLOR = (0, 255, 0, 255)
LOST_COLOR = (255, 0, 0, 255)


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


def clamp(value, min_value, max_value):
    if value < min_value:
        return min_value
    if value > max_value:
        return max_value
    return value


def safe_len(value):
    try:
        return len(value)
    except Exception:
        return 0


def safe_get(value, index, default=None):
    try:
        return value[index]
    except Exception:
        return default


def is_known_class(cls_id):
    if isinstance(LABELS, dict):
        return cls_id in LABELS
    return 0 <= cls_id < safe_len(LABELS)


def label_name(cls_id):
    if isinstance(LABELS, dict):
        return LABELS.get(cls_id, str(cls_id))
    if 0 <= cls_id < safe_len(LABELS):
        return LABELS[cls_id]
    return str(cls_id)


def draw_text(osd_img, x, y, text, color=TEXT_COLOR):
    try:
        osd_img.draw_string_advanced(x, y, FONT_SIZE, text, color=color)
    except Exception:
        try:
            osd_img.draw_string(x, y, text, color=color, scale=2)
        except Exception:
            pass


def draw_box(osd_img, x, y, w, h):
    x = int(x)
    y = int(y)
    w = int(w)
    h = int(h)
    try:
        osd_img.draw_rectangle(x, y, w, h, color=BOX_COLOR, thickness=3)
    except Exception:
        try:
            osd_img.draw_rectangle((x, y, w, h), color=BOX_COLOR, thickness=3)
        except Exception:
            pass


def clear_osd(osd_img):
    try:
        osd_img.clear()
    except Exception:
        pass


def normalize_score(score):
    # Some detectors return 0.95, others return 95. Convert both to 0.0~1.0.
    score = float(score)
    if score > 1.0:
        score = score / 100.0
    return score


def valid_box(x1, y1, x2, y2):
    return x2 > x1 and y2 > y1


def parse_flat_detection(det):
    # Fallback parser for flat YOLO boxes.
    # Output format is always: class_id, score, x, y, w, h.
    if det is None or len(det) < 6:
        return None

    # Common K230 format: [class_id, score, x1, y1, x2, y2]
    try:
        cls_id = int(det[0])
        score = normalize_score(det[1])
        x1 = float(det[2])
        y1 = float(det[3])
        x2 = float(det[4])
        y2 = float(det[5])
        if is_known_class(cls_id) and valid_box(x1, y1, x2, y2):
            return cls_id, score, x1, y1, x2 - x1, y2 - y1
    except Exception:
        pass

    # Another common format: [x1, y1, x2, y2, score, class_id]
    try:
        x1 = float(det[0])
        y1 = float(det[1])
        x2 = float(det[2])
        y2 = float(det[3])
        score = normalize_score(det[4])
        cls_id = int(det[5])
        if is_known_class(cls_id) and valid_box(x1, y1, x2, y2):
            return cls_id, score, x1, y1, x2 - x1, y2 - y1
    except Exception:
        pass

    return None


def choose_best_ball(dets):
    # Pick only one ball box with the highest confidence.
    # Main K230 YOLO result format:
    # dets[0] = boxes, each box is [x, y, w, h]
    # dets[1] = class ids
    # dets[2] = scores
    if dets is None or safe_len(dets) == 0:
        return None

    # K230 structured format:
    # dets[0] = boxes, dets[1] = class ids, dets[2] = scores.
    # Only enter this branch when dets[0][0] looks like a real box.
    first_box = safe_get(safe_get(dets, 0), 0)
    if safe_len(dets) >= 3 and safe_len(first_box) >= 4:
        boxes = dets[0]
        class_ids = dets[1]
        scores = dets[2]

        if boxes is not None and safe_len(boxes) > 0:
            best = None
            best_score = -1
            for i in range(safe_len(boxes)):
                box = boxes[i]
                if box is None or safe_len(box) < 4:
                    continue

                cls_id = 0
                if class_ids is not None and safe_len(class_ids) > i:
                    cls_id = int(class_ids[i])

                # This model has only one class: ball.
                if cls_id != BALL_CLASS_ID or not is_known_class(cls_id):
                    continue

                score = 1.0
                if scores is not None and safe_len(scores) > i:
                    score = normalize_score(scores[i])

                x = float(box[0])
                y = float(box[1])
                w = float(box[2])
                h = float(box[3])
                if w <= 0 or h <= 0:
                    continue

                if score > best_score:
                    best = cls_id, score, x, y, w, h
                    best_score = score
            if best is not None:
                return best

    # Some YOLO implementations return a single flat detection directly.
    info = parse_flat_detection(dets)
    if info is not None and info[0] == BALL_CLASS_ID:
        return info

    # Fallback for result formats that are already a list of flat detections.
    best = None
    best_score = -1
    for det in dets:
        info = parse_flat_detection(det)
        if info is None:
            continue
        cls_id, score, x, y, w, h = info
        if cls_id == BALL_CLASS_ID and score > best_score:
            best = info
            best_score = score
    return best


def ball_x_mm(ball_info):
    # Convert ball image center x from pixels to rod coordinate in millimeters.
    # Right side of center O is positive, left side is negative.
    cls_id, score, x, y, w, h = ball_info
    center_x_px = x + w * 0.5
    return (center_x_px - ROD_CENTER_PX) / PIXEL_PER_MM


def format_packet(x_mm, v_mm_s, confidence, state):
    # UART protocol: $B,+012,-034,095,1\n
    # x_mm and v_mm_s are signed 4 chars; confidence is 3 chars; state is 0 or 1.
    x_mm = int(clamp(round(x_mm), -999, 999))
    v_mm_s = int(clamp(round(v_mm_s), -999, 999))
    confidence = int(clamp(round(confidence), 0, 100))
    state = int(clamp(state, 0, 1))
    return "$B,%+04d,%+04d,%03d,%d\n" % (x_mm, v_mm_s, confidence, state)


def draw_ball_feedback(osd_img, ball_info, packet):
    if ball_info is None:
        draw_text(osd_img, 8, 8, "NO BALL", color=LOST_COLOR)
        draw_text(osd_img, 8, 40, packet.strip(), color=LOST_COLOR)
        return

    cls_id, score, x, y, w, h = ball_info
    draw_box(osd_img, x, y, w, h)
    text_y = int(y) - 28
    if text_y < 8:
        text_y = int(y) + 8
    draw_text(
        osd_img,
        int(x),
        text_y,
        "%s %.0f%%" % (label_name(cls_id), score * 100.0),
        color=TEXT_COLOR,
    )
    draw_text(osd_img, 8, 8, packet.strip(), color=TEXT_COLOR)


def main():
    print("K230 YOLOv8 ball model start")
    print("model:", KMODEL_PATH)
    print("labels:", LABELS)
    print("input:", MODEL_INPUT_SIZE)
    print("camera:", RGB888P_SIZE)
    print("display:", DISPLAY_SIZE, DISPLAY_MODE)
    print("conf/nms:", CONFIDENCE_THRESHOLD, NMS_THRESHOLD)
    print("uart:", UART_BAUDRATE)

    uart = YbUart(baudrate=UART_BAUDRATE)

    pl = PipeLine(
        rgb888p_size=RGB888P_SIZE,
        display_size=DISPLAY_SIZE,
        display_mode=DISPLAY_MODE,
    )

    if DISPLAY == "lcd2_4":
        pl.create(sensor=Sensor(id=2, width=SENSOR_SIZE[0], height=SENSOR_SIZE[1]))
    else:
        pl.create(sensor=Sensor(id=2, width=SENSOR_SIZE[0], height=SENSOR_SIZE[1]))

    display_size = pl.get_display_size()

    yolo = YOLOv8(
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
    last_x_mm = 0.0
    filtered_x_mm = 0.0
    last_detect_ms = ticks_ms()
    last_send_ms = 0

    try:
        while True:
            clock.tick()
            now_ms = ticks_ms()

            img = pl.get_frame()
            res = yolo.run(img)  # YOLO result list for the current frame.
            clear_osd(pl.osd_img)

            ball_info = choose_best_ball(res)  # None means the ball is lost.
            if ball_info is not None:
                cls_id, score, x, y, w, h = ball_info
                measured_x_mm = ball_x_mm(ball_info)  # Raw measured ball position.
                filtered_x_mm = (
                    (1.0 - POSITION_FILTER_ALPHA) * filtered_x_mm
                    + POSITION_FILTER_ALPHA * measured_x_mm
                )  # Filtered position sent to the car controller.

                dt_ms = ticks_diff(now_ms, last_detect_ms)
                if dt_ms > 0:
                    v_mm_s = (filtered_x_mm - last_x_mm) * 1000.0 / dt_ms  # Ball speed in mm/s.
                else:
                    v_mm_s = 0.0

                x_mm = filtered_x_mm
                confidence = score * 100.0
                state = 1
                last_x_mm = filtered_x_mm
                last_detect_ms = now_ms
            else:
                # Lost ball packet. The car controller should stop using old vision data.
                x_mm = 0
                v_mm_s = 0
                confidence = 0
                state = 0

            packet = format_packet(x_mm, v_mm_s, confidence, state)  # Final UART frame.
            if ticks_diff(now_ms, last_send_ms) >= UART_SEND_INTERVAL_MS:
                uart.send(packet)  # Send motion data to the car controller.
                last_send_ms = now_ms

            yolo.draw_result(res, pl.osd_img)  # Keep the library LCD detection overlay.
            draw_ball_feedback(pl.osd_img, ball_info, packet)
            print(res)  # Debug: raw YOLO result. Use this to verify box order.
            print(packet)  # Debug: exact UART payload.
            pl.show_image()
            print("FPS:", clock.fps())
            gc.collect()
    except Exception as e:
        print("YOLOv8 stopped:", e)
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
