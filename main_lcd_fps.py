import gc
import time

from libs.PipeLine import PipeLine
from libs.YOLO import YOLO11
from media.sensor import Sensor


# New model package from 9476.tar.
KMODEL_PATH = "/sdcard/yolo11n_det_320.kmodel"
LABELS = {0: "ball"}
MODEL_INPUT_SIZE = [320, 320]

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

CONFIDENCE_THRESHOLD = 0.60
NMS_THRESHOLD = 0.45

FPS_X = 8
FPS_Y = 8
FPS_FONT_SIZE = 28
FPS_TEXT_COLOR = (255, 255, 0, 255)


def draw_fps(osd_img, fps):
    text = "FPS: {:.1f}".format(fps)
    try:
        osd_img.draw_string_advanced(
            FPS_X,
            FPS_Y,
            FPS_FONT_SIZE,
            text,
            color=FPS_TEXT_COLOR,
        )
    except AttributeError:
        osd_img.draw_string(
            FPS_X,
            FPS_Y,
            text,
            color=FPS_TEXT_COLOR,
            scale=2,
        )


def main():
    print("K230 YOLO11 new model start")
    print("model:", KMODEL_PATH)
    print("labels:", LABELS)
    print("input:", MODEL_INPUT_SIZE)
    print("camera:", RGB888P_SIZE)
    print("display:", DISPLAY_SIZE, DISPLAY_MODE)

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

    try:
        while True:
            clock.tick()
            img = pl.get_frame()
            res = yolo.run(img)
            yolo.draw_result(res, pl.osd_img)
            fps = clock.fps()
            draw_fps(pl.osd_img, fps)
            print(res)
            pl.show_image()
            print("FPS:", fps)
            gc.collect()
    except Exception as e:
        print("YOLO11 stopped:", e)
    finally:
        try:
            yolo.deinit()
        except Exception:
            pass
        try:
            pl.destroy()
        except Exception:
            pass
        gc.collect()


if __name__ == "__main__":
    main()
