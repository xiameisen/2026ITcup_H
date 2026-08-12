import gc
import time

from libs.PipeLine import PipeLine
from libs.YOLO import YOLO11
from media.sensor import Sensor
from ybUtils.YbUart import YbUart


# -------------------- Model --------------------
KMODEL_PATH = "/sdcard/yolo11n_det_320.kmodel"
LABELS = {0: "steel ball"}
BALL_CLASS_ID = 0
MODEL_INPUT_SIZE = [320, 320]


# -------------------- Display and camera --------------------
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


# -------------------- Detection --------------------
CONFIDENCE_THRESHOLD = 0.45
NMS_THRESHOLD = 0.45
MAX_BOXES = 50

# Reject boxes that are far from the tube axis or have unreasonable size.
# Tune these after the camera is fixed.
TRACK_HALF_WIDTH_PX = 120.0
BALL_MIN_BOX_PX = 6.0
BALL_MAX_BOX_PX = 110.0


# -------------------- Axis calibration --------------------
# Camera is above the tube. Tune these three points on the 640x480 IDE image:
# AXIS_LEFT_PX   = pixel point near the -11.5 cm mark
# AXIS_CENTER_PX = pixel point near the 0 cm mark
# AXIS_RIGHT_PX  = pixel point near the +11.5 cm mark
# Initial values estimated from the current top-down screen view.
# Fine tune them after checking the overlay on the K230 IDE frame buffer.
AXIS_LEFT_PX = (26.0, 252.0)
AXIS_CENTER_PX = (350.8, 300.0)
AXIS_RIGHT_PX = (620.0, 252.0)
HALF_LENGTH_MM = 115.0


# -------------------- Motion filter --------------------
# Larger alpha = faster response, smaller alpha = smoother output.
POSITION_ALPHA = 0.35
VELOCITY_ALPHA = 0.25
ACCEL_ALPHA = 0.20


# -------------------- UART --------------------
UART_BAUDRATE = 115200
UART_SEND_INTERVAL_MS = 50  # 20 Hz output.


# -------------------- Debug overlay --------------------
FONT_SIZE = 20
TEXT_COLOR = (255, 255, 0, 255)
AXIS_COLOR = (0, 255, 255, 255)
BOX_COLOR = (0, 255, 0, 255)
LOST_COLOR = (255, 0, 0, 255)
DEBUG_PRINT = True
PRINT_INTERVAL_MS = 1000
PROFILE_TIMING = False
USE_YOLO_LIBRARY_DRAW = False
ENABLE_IDE_PREVIEW = True
PREVIEW_EVERY_N_FRAMES = 2
GC_EVERY_N_FRAMES = 5


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


def normalize_score(score):
    score = float(score)
    if score > 1.0:
        score = score / 100.0
    return score


def known_class(cls_id):
    if isinstance(LABELS, dict):
        return cls_id in LABELS
    return 0 <= cls_id < safe_len(LABELS)


def label_name(cls_id):
    if isinstance(LABELS, dict):
        return LABELS.get(cls_id, str(cls_id))
    if 0 <= cls_id < safe_len(LABELS):
        return LABELS[cls_id]
    return str(cls_id)


def axis_params():
    dx = AXIS_RIGHT_PX[0] - AXIS_LEFT_PX[0]
    dy = AXIS_RIGHT_PX[1] - AXIS_LEFT_PX[1]
    axis_len_px = (dx * dx + dy * dy) ** 0.5
    if axis_len_px <= 1.0:
        axis_len_px = 1.0
    ux = dx / axis_len_px
    uy = dy / axis_len_px
    mm_per_px = (HALF_LENGTH_MM * 2.0) / axis_len_px
    return ux, uy, mm_per_px, axis_len_px


def project_to_axis(point):
    ux, uy, mm_per_px, axis_len_px = axis_params()
    rel_x = float(point[0]) - AXIS_CENTER_PX[0]
    rel_y = float(point[1]) - AXIS_CENTER_PX[1]
    along_px = rel_x * ux + rel_y * uy
    cross_px = rel_x * (-uy) + rel_y * ux
    if cross_px < 0:
        cross_px = -cross_px
    return along_px * mm_per_px, cross_px, along_px


def draw_text(osd_img, x, y, text, color=TEXT_COLOR):
    try:
        osd_img.draw_string_advanced(int(x), int(y), FONT_SIZE, text, color=color)
    except Exception:
        try:
            osd_img.draw_string(int(x), int(y), text, color=color, scale=2)
        except Exception:
            pass


def draw_line(osd_img, x1, y1, x2, y2, color=AXIS_COLOR, thickness=2):
    try:
        osd_img.draw_line(int(x1), int(y1), int(x2), int(y2), color=color, thickness=thickness)
    except Exception:
        try:
            osd_img.draw_line((int(x1), int(y1), int(x2), int(y2)), color=color, thickness=thickness)
        except Exception:
            pass


def draw_rectangle(osd_img, x, y, w, h, color=BOX_COLOR, thickness=3):
    try:
        osd_img.draw_rectangle(int(x), int(y), int(w), int(h), color=color, thickness=thickness)
    except Exception:
        try:
            osd_img.draw_rectangle((int(x), int(y), int(w), int(h)), color=color, thickness=thickness)
        except Exception:
            pass


def clear_osd(osd_img):
    try:
        osd_img.clear()
    except Exception:
        pass


def draw_cross(osd_img, x, y, size, color=AXIS_COLOR):
    draw_line(osd_img, x - size, y, x + size, y, color=color, thickness=2)
    draw_line(osd_img, x, y - size, x, y + size, color=color, thickness=2)


def draw_axis_overlay(osd_img):
    draw_line(
        osd_img,
        AXIS_LEFT_PX[0],
        AXIS_LEFT_PX[1],
        AXIS_RIGHT_PX[0],
        AXIS_RIGHT_PX[1],
        color=AXIS_COLOR,
        thickness=4,
    )
    draw_cross(osd_img, AXIS_LEFT_PX[0], AXIS_LEFT_PX[1], 16, color=AXIS_COLOR)
    draw_cross(osd_img, AXIS_CENTER_PX[0], AXIS_CENTER_PX[1], 18, color=LOST_COLOR)
    draw_cross(osd_img, AXIS_RIGHT_PX[0], AXIS_RIGHT_PX[1], 16, color=AXIS_COLOR)
    draw_text(osd_img, AXIS_LEFT_PX[0] + 6, AXIS_LEFT_PX[1] - 28, "L -11.5", color=AXIS_COLOR)
    draw_text(osd_img, AXIS_CENTER_PX[0] + 8, AXIS_CENTER_PX[1] + 8, "O 0", color=LOST_COLOR)
    draw_text(osd_img, AXIS_RIGHT_PX[0] - 96, AXIS_RIGHT_PX[1] - 28, "R +11.5", color=AXIS_COLOR)


def valid_xywh(x, y, w, h):
    if w <= BALL_MIN_BOX_PX or h <= BALL_MIN_BOX_PX:
        return False
    if w >= BALL_MAX_BOX_PX or h >= BALL_MAX_BOX_PX:
        return False

    center = (x + w * 0.5, y + h * 0.5)
    x_mm, cross_px, along_px = project_to_axis(center)
    if cross_px > TRACK_HALF_WIDTH_PX:
        return False
    if x_mm < -HALF_LENGTH_MM - 30.0 or x_mm > HALF_LENGTH_MM + 30.0:
        return False
    return True


def make_ball_info(cls_id, score, x, y, a, b):
    if cls_id != BALL_CLASS_ID or not known_class(cls_id):
        return None

    score = normalize_score(score)
    if score < CONFIDENCE_THRESHOLD:
        return None

    x = float(x)
    y = float(y)
    a = float(a)
    b = float(b)

    # Candidate 1: [x, y, w, h].
    if valid_xywh(x, y, a, b):
        return cls_id, score, x, y, a, b

    # Candidate 2: [x1, y1, x2, y2].
    w = a - x
    h = b - y
    if valid_xywh(x, y, w, h):
        return cls_id, score, x, y, w, h

    return None


def parse_flat_detection(det):
    if det is None or safe_len(det) < 6:
        return None

    # Format A: [class_id, score, x, y, w_or_x2, h_or_y2].
    try:
        info = make_ball_info(int(det[0]), det[1], det[2], det[3], det[4], det[5])
        if info is not None:
            return info
    except Exception:
        pass

    # Format B: [x, y, w_or_x2, h_or_y2, score, class_id].
    try:
        info = make_ball_info(int(det[5]), det[4], det[0], det[1], det[2], det[3])
        if info is not None:
            return info
    except Exception:
        pass

    return None


def choose_best_ball(dets):
    if dets is None or safe_len(dets) == 0:
        return None

    best = None
    best_score = -1.0

    # K230 structured format:
    # dets[0] = boxes, dets[1] = class ids, dets[2] = scores.
    first_box = safe_get(safe_get(dets, 0), 0)
    if safe_len(dets) >= 3 and safe_len(first_box) >= 4:
        boxes = dets[0]
        class_ids = dets[1]
        scores = dets[2]
        for i in range(safe_len(boxes)):
            box = safe_get(boxes, i)
            if box is None or safe_len(box) < 4:
                continue
            cls_id = BALL_CLASS_ID
            if class_ids is not None and safe_len(class_ids) > i:
                cls_id = int(class_ids[i])
            score = 1.0
            if scores is not None and safe_len(scores) > i:
                score = scores[i]
            info = make_ball_info(cls_id, score, box[0], box[1], box[2], box[3])
            if info is not None and info[1] > best_score:
                best = info
                best_score = info[1]
        if best is not None:
            return best

    # Single flat detection.
    info = parse_flat_detection(dets)
    if info is not None:
        return info

    # List of flat detections.
    for det in dets:
        info = parse_flat_detection(det)
        if info is None:
            continue
        if info[1] > best_score:
            best = info
            best_score = info[1]

    return best


def ball_center(ball_info):
    cls_id, score, x, y, w, h = ball_info
    return x + w * 0.5, y + h * 0.5


def ball_x_mm(ball_info):
    center = ball_center(ball_info)
    x_mm, cross_px, along_px = project_to_axis(center)
    return x_mm, cross_px


def low_pass(old_value, new_value, alpha):
    return (1.0 - alpha) * old_value + alpha * new_value


def format_packet(x_mm, v_mm_s, a_mm_s2, confidence, state):
    x_mm = int(clamp(round(x_mm), -9999, 9999))
    v_mm_s = int(clamp(round(v_mm_s), -9999, 9999))
    a_mm_s2 = int(clamp(round(a_mm_s2), -99999, 99999))
    confidence = int(clamp(round(confidence), 0, 100))
    state = int(clamp(state, 0, 1))
    return "$B,%+05d,%+05d,%+06d,%03d,%d\n" % (
        x_mm,
        v_mm_s,
        a_mm_s2,
        confidence,
        state,
    )


def draw_ball_overlay(osd_img, ball_info, x_mm, v_mm_s, a_mm_s2, confidence, state):
    draw_axis_overlay(osd_img)

    if state == 0 or ball_info is None:
        draw_text(osd_img, 8, 8, "x:----cm", color=LOST_COLOR)
        return

    cls_id, score, x, y, w, h = ball_info
    draw_rectangle(osd_img, x, y, w, h, color=BOX_COLOR, thickness=3)
    center_x, center_y = ball_center(ball_info)
    draw_cross(osd_img, center_x, center_y, 8, color=BOX_COLOR)
    label_y = y - 24
    if label_y < 8:
        label_y = y + h + 4
    draw_text(
        osd_img,
        x,
        label_y,
        "%s %.0f%%" % (label_name(cls_id), score * 100.0),
        color=TEXT_COLOR,
    )
    draw_text(osd_img, 8, 8, "x:%+.1fcm" % (x_mm / 10.0), color=TEXT_COLOR)


def main():
    print("K230 ball control vision v2 start")
    print("model:", KMODEL_PATH)
    print("labels:", LABELS)
    print("input:", MODEL_INPUT_SIZE)
    print("camera:", RGB888P_SIZE)
    print("display:", DISPLAY_SIZE, DISPLAY_MODE)
    print("axis left/center/right:", AXIS_LEFT_PX, AXIS_CENTER_PX, AXIS_RIGHT_PX)
    print("half length mm:", HALF_LENGTH_MM)
    print("uart:", UART_BAUDRATE)
    print("packet: $B,x_mm,v_mm_s,a_mm_s2,conf,state")

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
        max_boxes_num=MAX_BOXES,
        debug_mode=0,
    )
    yolo.config_preprocess()

    clock = time.clock()
    last_send_ms = 0
    last_print_ms = 0
    last_motion_ms = ticks_ms()
    frame_count = 0
    prof_frames = 0
    prof_get_ms = 0
    prof_run_ms = 0
    prof_post_ms = 0
    prof_draw_ms = 0
    prof_show_ms = 0

    filter_ready = False
    x_filter = 0.0
    v_filter = 0.0
    a_filter = 0.0

    try:
        while True:
            clock.tick()
            now_ms = ticks_ms()
            frame_count += 1

            t0 = ticks_ms()
            img = pl.get_frame()
            t1 = ticks_ms()
            res = yolo.run(img)
            t2 = ticks_ms()
            ball_info = choose_best_ball(res)

            if ball_info is not None:
                raw_x_mm, cross_px = ball_x_mm(ball_info)
                dt_ms = ticks_diff(now_ms, last_motion_ms)
                if dt_ms <= 0:
                    dt_ms = 1
                dt_s = dt_ms / 1000.0

                if not filter_ready:
                    x_filter = raw_x_mm
                    v_filter = 0.0
                    a_filter = 0.0
                    filter_ready = True
                else:
                    prev_x = x_filter
                    prev_v = v_filter
                    x_filter = low_pass(x_filter, raw_x_mm, POSITION_ALPHA)
                    raw_v = (x_filter - prev_x) / dt_s
                    v_filter = low_pass(v_filter, raw_v, VELOCITY_ALPHA)
                    raw_a = (v_filter - prev_v) / dt_s
                    a_filter = low_pass(a_filter, raw_a, ACCEL_ALPHA)

                last_motion_ms = now_ms
                confidence = ball_info[1] * 100.0
                state = 1
                out_x = x_filter
                out_v = v_filter
                out_a = a_filter
            else:
                confidence = 0
                state = 0
                out_x = 0.0
                out_v = 0.0
                out_a = 0.0
            t3 = ticks_ms()

            packet = format_packet(out_x, out_v, out_a, confidence, state)
            if ticks_diff(now_ms, last_send_ms) >= UART_SEND_INTERVAL_MS:
                uart.send(packet)
                last_send_ms = now_ms

            do_preview = ENABLE_IDE_PREVIEW and frame_count % PREVIEW_EVERY_N_FRAMES == 0
            if do_preview:
                clear_osd(pl.osd_img)
                if USE_YOLO_LIBRARY_DRAW:
                    yolo.draw_result(res, pl.osd_img)
                draw_ball_overlay(pl.osd_img, ball_info, out_x, out_v, out_a, confidence, state)
            t4 = ticks_ms()

            if do_preview:
                pl.show_image()
            t5 = ticks_ms()

            if PROFILE_TIMING:
                prof_frames += 1
                prof_get_ms += ticks_diff(t1, t0)
                prof_run_ms += ticks_diff(t2, t1)
                prof_post_ms += ticks_diff(t3, t2)
                prof_draw_ms += ticks_diff(t4, t3)
                prof_show_ms += ticks_diff(t5, t4)

            if DEBUG_PRINT and ticks_diff(now_ms, last_print_ms) >= PRINT_INTERVAL_MS:
                if PROFILE_TIMING and prof_frames > 0:
                    print(
                        "FPS: %.1f x:%+.1fcm state:%d get:%d run:%d post:%d draw:%d show:%d"
                        % (
                            clock.fps(),
                            out_x / 10.0,
                            state,
                            prof_get_ms // prof_frames,
                            prof_run_ms // prof_frames,
                            prof_post_ms // prof_frames,
                            prof_draw_ms // prof_frames,
                            prof_show_ms // prof_frames,
                        )
                    )
                    prof_frames = 0
                    prof_get_ms = 0
                    prof_run_ms = 0
                    prof_post_ms = 0
                    prof_draw_ms = 0
                    prof_show_ms = 0
                else:
                    print("FPS: %.1f x:%+.1fcm state:%d" % (clock.fps(), out_x / 10.0, state))
                last_print_ms = now_ms

            if frame_count % GC_EVERY_N_FRAMES == 0:
                gc.collect()
    except Exception as e:
        print("vision v2 stopped:", e)
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
