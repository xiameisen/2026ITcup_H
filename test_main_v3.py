import gc
import time

try:
    from machine import TOUCH
except Exception:
    TOUCH = None

from libs.PipeLine import PipeLine
from libs.YOLO import YOLOv8
from media.sensor import Sensor
from ybUtils.YbUart import YbUart


# -------------------- Model --------------------
KMODEL_PATH = "/sdcard/model.kmodel"
LABELS = ["ball"]
BALL_CLASS_ID = 0
MODEL_INPUT_SIZE = [640, 640]


# -------------------- Display and camera --------------------
# Keep lcd2_4 if the current IDE/LCD view is 640x480.
# If your physical touch screen is the 3.5 inch MIPI screen, try "lcd3_5".
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
CONFIDENCE_THRESHOLD = 0.50
NMS_THRESHOLD = 0.50
MAX_BOXES = 50

# Reject boxes that are far from the tube axis or have unreasonable size.
TRACK_HALF_WIDTH_PX = 120.0
BALL_MIN_BOX_PX = 6.0
BALL_MAX_BOX_PX = 110.0


# -------------------- Touch calibration --------------------
TOUCH_ENABLE = True
TOUCH_CALIBRATION_ON_START = True
LOAD_CALIBRATION_ON_START = True
SAVE_TOUCH_CALIBRATION = True
CALIBRATION_FILE = "/sdcard/ball_axis_calib.txt"
CALIBRATION_TIMEOUT_MS = 120000

# Use these only if touch coordinates are rotated/flipped relative to the LCD.
TOUCH_SWAP_XY = False
TOUCH_FLIP_X = False
TOUCH_FLIP_Y = False
TOUCH_RAW_WIDTH = 0
TOUCH_RAW_HEIGHT = 0
TOUCH_TAP_DEBOUNCE_MS = 350
TOUCH_NEW_TAP_DISTANCE_PX = 25.0


# -------------------- Axis calibration fallback --------------------
# These are used before touch calibration, or when touch is unavailable.
AXIS_LEFT_PX = (70.0, 392.0)
AXIS_CENTER_PX = (320.0, 392.0)
AXIS_RIGHT_PX = (570.0, 392.0)
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
OK_COLOR = (0, 255, 0, 255)
PRINT_INTERVAL_MS = 200


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


def sleep_ms(ms):
    try:
        time.sleep_ms(ms)
    except AttributeError:
        time.sleep(ms / 1000.0)


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
        thickness=2,
    )
    draw_cross(osd_img, AXIS_LEFT_PX[0], AXIS_LEFT_PX[1], 8, color=AXIS_COLOR)
    draw_cross(osd_img, AXIS_CENTER_PX[0], AXIS_CENTER_PX[1], 12, color=LOST_COLOR)
    draw_cross(osd_img, AXIS_RIGHT_PX[0], AXIS_RIGHT_PX[1], 8, color=AXIS_COLOR)
    draw_text(osd_img, AXIS_CENTER_PX[0] + 8, AXIS_CENTER_PX[1] + 8, "O", color=LOST_COLOR)


def draw_selected_points(osd_img, points):
    labels = ["L", "O", "R"]
    colors = [AXIS_COLOR, LOST_COLOR, AXIS_COLOR]
    for i in range(3):
        point = points[i]
        if point is None:
            continue
        draw_cross(osd_img, point[0], point[1], 10, color=colors[i])
        draw_text(osd_img, point[0] + 8, point[1] + 8, labels[i], color=colors[i])
    if points[0] is not None and points[2] is not None:
        draw_line(osd_img, points[0][0], points[0][1], points[2][0], points[2][1])


def midpoint(left, right):
    return ((float(left[0]) + float(right[0])) * 0.5, (float(left[1]) + float(right[1])) * 0.5)


def init_touch():
    if not TOUCH_ENABLE:
        print("[TOUCH] disabled")
        return None
    if TOUCH is None:
        print("[TOUCH] TOUCH class not available")
        return None
    try:
        touch = TOUCH(0)
        print("[TOUCH] init ok")
        return touch
    except Exception as error:
        print("[TOUCH] init failed:", error)
        return None


def extract_touch_xy(point):
    x = None
    y = None
    try:
        x = point.x
        y = point.y
    except Exception:
        pass
    if x is None or y is None:
        try:
            x = point[0]
            y = point[1]
        except Exception:
            return None
    return float(x), float(y)


def extract_touch_event(point):
    try:
        return point.event
    except Exception:
        pass
    try:
        return point[2]
    except Exception:
        return None


def touch_const(name):
    if TOUCH is None:
        return None
    try:
        return getattr(TOUCH, name)
    except Exception:
        return None


def touch_event_is_down(event):
    if event is None:
        return True
    down = touch_const("EVENT_DOWN")
    if down is not None and event == down:
        return True
    # Some Yahboom/CanMV examples report the first touch as event 0.
    return event == 0


def touch_event_is_up(event):
    if event is None:
        return False
    up = touch_const("EVENT_UP")
    if up is not None and event == up:
        return True
    release = touch_const("EVENT_RELEASE")
    if release is not None and event == release:
        return True
    return False


def point_distance(a, b):
    if a is None or b is None:
        return 99999.0
    dx = float(a[0]) - float(b[0])
    dy = float(a[1]) - float(b[1])
    return (dx * dx + dy * dy) ** 0.5


def map_touch_to_display(x, y):
    if TOUCH_RAW_WIDTH > 0 and TOUCH_RAW_HEIGHT > 0:
        x = x * DISPLAY_SIZE[0] / float(TOUCH_RAW_WIDTH)
        y = y * DISPLAY_SIZE[1] / float(TOUCH_RAW_HEIGHT)

    if TOUCH_SWAP_XY:
        x, y = y, x
    if TOUCH_FLIP_X:
        x = DISPLAY_SIZE[0] - 1 - x
    if TOUCH_FLIP_Y:
        y = DISPLAY_SIZE[1] - 1 - y

    x = clamp(x, 0.0, DISPLAY_SIZE[0] - 1.0)
    y = clamp(y, 0.0, DISPLAY_SIZE[1] - 1.0)
    return x, y


def read_touch_sample(touch):
    if touch is None:
        return None
    try:
        points = touch.read(1)
    except Exception:
        return None
    if points is None or safe_len(points) <= 0:
        return None
    xy = extract_touch_xy(points[0])
    if xy is None:
        return None
    x, y = map_touch_to_display(xy[0], xy[1])
    return x, y, extract_touch_event(points[0])


def set_axis_points(left, center, right):
    global AXIS_LEFT_PX
    global AXIS_CENTER_PX
    global AXIS_RIGHT_PX
    AXIS_LEFT_PX = (float(left[0]), float(left[1]))
    AXIS_CENTER_PX = (float(center[0]), float(center[1]))
    AXIS_RIGHT_PX = (float(right[0]), float(right[1]))


def save_calibration():
    if not SAVE_TOUCH_CALIBRATION:
        return False
    try:
        with open(CALIBRATION_FILE, "w") as f:
            f.write("%.2f,%.2f\n" % (AXIS_LEFT_PX[0], AXIS_LEFT_PX[1]))
            f.write("%.2f,%.2f\n" % (AXIS_CENTER_PX[0], AXIS_CENTER_PX[1]))
            f.write("%.2f,%.2f\n" % (AXIS_RIGHT_PX[0], AXIS_RIGHT_PX[1]))
            f.write("%.2f\n" % HALF_LENGTH_MM)
        print("[CAL] saved:", CALIBRATION_FILE)
        return True
    except Exception as error:
        print("[CAL] save failed:", error)
        return False


def parse_pair(line):
    parts = line.strip().split(",")
    if len(parts) < 2:
        return None
    return float(parts[0]), float(parts[1])


def load_calibration():
    global HALF_LENGTH_MM
    if not LOAD_CALIBRATION_ON_START:
        return False
    try:
        with open(CALIBRATION_FILE, "r") as f:
            lines = f.readlines()
        if len(lines) < 3:
            return False
        left = parse_pair(lines[0])
        center = parse_pair(lines[1])
        right = parse_pair(lines[2])
        if left is None or center is None or right is None:
            return False
        set_axis_points(left, center, right)
        if len(lines) >= 4:
            HALF_LENGTH_MM = float(lines[3].strip())
        print("[CAL] loaded:", AXIS_LEFT_PX, AXIS_CENTER_PX, AXIS_RIGHT_PX)
        return True
    except Exception:
        return False


def run_touch_calibration(pl, touch):
    if touch is None:
        return False

    step_names = ["LEFT -11.5cm", "RIGHT +11.5cm"]
    points = [None, None, None]
    step = 0
    ready_for_tap = True
    last_tap_point = None
    last_tap_ms = -10000
    start_ms = ticks_ms()

    print("[CAL] touch the marks: left, right. Center is calculated automatically.")
    while step < 2:
        now_ms = ticks_ms()
        if ticks_diff(now_ms, start_ms) > CALIBRATION_TIMEOUT_MS:
            print("[CAL] timeout, using previous/default axis")
            return False

        pl.get_frame()
        clear_osd(pl.osd_img)
        draw_selected_points(pl.osd_img, points)
        draw_text(pl.osd_img, 8, 8, "TOUCH CALIBRATION", color=TEXT_COLOR)
        draw_text(pl.osd_img, 8, 36, "Tap: %s" % step_names[step], color=TEXT_COLOR)
        draw_text(pl.osd_img, 8, 64, "Step %d/2" % (step + 1), color=TEXT_COLOR)
        draw_text(pl.osd_img, 8, 92, "Center O = midpoint", color=TEXT_COLOR)

        sample = read_touch_sample(touch)
        if sample is None:
            ready_for_tap = True
        else:
            point = (sample[0], sample[1])
            event = sample[2]
            draw_cross(pl.osd_img, point[0], point[1], 14, color=OK_COLOR)
            draw_text(pl.osd_img, 8, 120, "touch event: %s" % str(event), color=TEXT_COLOR)

            if touch_event_is_up(event):
                ready_for_tap = True

            enough_time = ticks_diff(now_ms, last_tap_ms) >= TOUCH_TAP_DEBOUNCE_MS
            moved_to_new_mark = (
                point_distance(point, last_tap_point) >= TOUCH_NEW_TAP_DISTANCE_PX
            )
            is_new_tap = (
                touch_event_is_down(event) and ready_for_tap
            ) or (
                enough_time and moved_to_new_mark
            )

            if is_new_tap:
                if step == 0:
                    points[0] = point
                else:
                    points[2] = point
                    points[1] = midpoint(points[0], points[2])
                print("[CAL]", step_names[step], point, "event:", event)
                step += 1
                ready_for_tap = False
                last_tap_point = point
                last_tap_ms = now_ms
                sleep_ms(250)

        pl.show_image()
        gc.collect()

    if points[1] is None:
        points[1] = midpoint(points[0], points[2])
    set_axis_points(points[0], points[1], points[2])
    save_calibration()

    end_ms = ticks_ms()
    while ticks_diff(ticks_ms(), end_ms) < 900:
        pl.get_frame()
        clear_osd(pl.osd_img)
        draw_axis_overlay(pl.osd_img)
        draw_text(pl.osd_img, 8, 8, "CALIBRATION OK", color=OK_COLOR)
        draw_text(pl.osd_img, 8, 36, "O is midpoint", color=OK_COLOR)
        draw_text(pl.osd_img, 8, 64, "Start vision...", color=OK_COLOR)
        pl.show_image()
        gc.collect()

    return True


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
        draw_text(osd_img, 8, 8, "NO BALL", color=LOST_COLOR)
        draw_text(osd_img, 8, 34, "x:+0.0cm v:+0.0cm/s", color=LOST_COLOR)
        draw_text(osd_img, 8, 60, "a:+0.0cm/s2 c:000%", color=LOST_COLOR)
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
    draw_text(osd_img, 8, 8, "x:%+.1fcm v:%+.1fcm/s" % (x_mm / 10.0, v_mm_s / 10.0))
    draw_text(osd_img, 8, 34, "a:%+.1fcm/s2 c:%03d%%" % (a_mm_s2 / 10.0, int(confidence)))


def main():
    print("K230 ball control vision v3 start")
    print("model:", KMODEL_PATH)
    print("labels:", LABELS)
    print("input:", MODEL_INPUT_SIZE)
    print("camera:", RGB888P_SIZE)
    print("display:", DISPLAY_SIZE, DISPLAY_MODE)
    print("half length mm:", HALF_LENGTH_MM)
    print("uart:", UART_BAUDRATE)
    print("packet: $B,x_mm,v_mm_s,a_mm_s2,conf,state")

    load_calibration()
    touch = init_touch()

    uart = None
    pl = None
    yolo = None

    try:
        pl = PipeLine(
            rgb888p_size=RGB888P_SIZE,
            display_size=DISPLAY_SIZE,
            display_mode=DISPLAY_MODE,
        )
        pl.create(sensor=Sensor(id=2, width=SENSOR_SIZE[0], height=SENSOR_SIZE[1]))
        display_size = pl.get_display_size()
        try:
            DISPLAY_SIZE[0] = display_size[0]
            DISPLAY_SIZE[1] = display_size[1]
        except Exception:
            pass
        print("display actual:", display_size)

        if TOUCH_CALIBRATION_ON_START:
            if not run_touch_calibration(pl, touch):
                print("[CAL] axis:", AXIS_LEFT_PX, AXIS_CENTER_PX, AXIS_RIGHT_PX)

        uart = YbUart(baudrate=UART_BAUDRATE)

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
            max_boxes_num=MAX_BOXES,
            debug_mode=0,
        )
        yolo.config_preprocess()

        clock = time.clock()
        last_send_ms = 0
        last_print_ms = 0
        last_motion_ms = ticks_ms()

        filter_ready = False
        x_filter = 0.0
        v_filter = 0.0
        a_filter = 0.0

        while True:
            clock.tick()
            now_ms = ticks_ms()

            img = pl.get_frame()
            res = yolo.run(img)
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

            packet = format_packet(out_x, out_v, out_a, confidence, state)
            if ticks_diff(now_ms, last_send_ms) >= UART_SEND_INTERVAL_MS:
                uart.send(packet)
                last_send_ms = now_ms

            clear_osd(pl.osd_img)
            yolo.draw_result(res, pl.osd_img)
            draw_ball_overlay(pl.osd_img, ball_info, out_x, out_v, out_a, confidence, state)

            if ticks_diff(now_ms, last_print_ms) >= PRINT_INTERVAL_MS:
                print(packet.strip())
                print("ball:", ball_info)
                print("FPS:", clock.fps())
                last_print_ms = now_ms

            pl.show_image()
            gc.collect()
    except Exception as e:
        print("vision v3 stopped:", e)
    finally:
        try:
            if yolo is not None:
                yolo.deinit()
        except Exception:
            pass
        try:
            if pl is not None:
                pl.destroy()
        except Exception:
            pass
        try:
            if uart is not None:
                uart.deinit()
        except Exception:
            pass
        gc.collect()


if __name__ == "__main__":
    main()
