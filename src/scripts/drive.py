import argparse
import logging
import os
import time

import cv2
from jtop import jtop
from mpu9250_jmdev.registers import (
                        AK8963_ADDRESS,
                        MPU9050_ADDRESS_68,
                        GFS_1000,
                        AFS_8G,
                        AK8963_BIT_16,
                        AK8963_MODE_C100HZ
                    )
from mpu9250_jmdev.mpu_9250 import MPU9250
import torch
import wandb
import yolov5
from yolov5.utils.general import non_max_suppression

from jetcam.csi_camera import CSICamera
from jetracer.nvidia_racecar import NvidiaRacecar
from wandb_jetracer.utils.xy_dataset import preprocess
from wandb_jetracer.utils.utils import setup_logging, show_label
from torch2trt import TRTModule

THROTTLE_GAIN = -1
STEERING_GAIN = -2  # TODO: add that to the config
IMG_SIZE = 224


def setup(config):

    model_trt = TRTModule()

    if config.local_model is None:
        logging.info("Downloading latest optimized model...")
        artifact = wandb.use_artifact(f'trt-model:{config.model_version}')
        artifact_dir = artifact.download()
        model_trt.load_state_dict(torch.load(
            os.path.join(artifact_dir, 'trt-model.pth')
        ))
    else:
        logging.info(f"Using local model: {config.local_model}")
        model_trt.load_state_dict(torch.load(config.local_model))

    yolo_model = None
    if config.yolo:
        logging.info("Setting up yolo model")
        yolo_model = yolov5.load('yolov5s.pt')
        yolo_model.half()

    logging.info("Setting up car and camera")
    car = NvidiaRacecar()
    camera = CSICamera(
        width=IMG_SIZE, height=IMG_SIZE, capture_fps=config.framerate
    )

    logging.info("Setting up MPU9250")
    mpu = MPU9250(
        address_ak=AK8963_ADDRESS,
        address_mpu_master=MPU9050_ADDRESS_68,  # In 0x68 Address
        address_mpu_slave=None,
        bus=0,  # TODO set that in config file
        gfs=GFS_1000,
        afs=AFS_8G,
        mfs=AK8963_BIT_16,
        mode=AK8963_MODE_C100HZ)

    mpu.configure()  # Apply the settings to the registers.

    return car, camera, mpu, model_trt, yolo_model


def control_policy(road_center, objects, config):
    x, y = road_center
    steering = x * STEERING_GAIN  # *(y+1)/2
    throttle = config.throttle * THROTTLE_GAIN

    return throttle, steering


def infer(image, model_trt, yolo_model=None):
    image = preprocess(image).half()

    objects = None
    if yolo_model is not None:
        objects = yolo_model(image, size=IMG_SIZE)[0]
        objects = non_max_suppression(
            objects,
            yolo_model.conf,
            iou_thres=yolo_model.iou
        )

    output = model_trt(image).squeeze()  # .detach().cpu().numpy().flatten()
    x, y = float(output[0]), float(output[1])

    return (x, y), objects


def format_jetson_stats(stats):
    s_metrics = ["GPU", "Temp GPU", "Temp CPU", "power avg", "power cur"]
    system_stats = {k: stats[k] for k in s_metrics}

    return system_stats


def read_mpu(mpu):
    accel_x, accel_y, accel_z = mpu.readAccelerometerMaster()
    gyro_x, gyro_y, gyro_z = mpu.readGyroscopeMaster()
    magneto_x, magneto_y, magneto_z = mpu.readMagnetometerMaster()

    # TODO: doesn't feel very pythonic, fix that
    return {
        "car/accelerometer_x": accel_x,
        "car/accelerometer_y": accel_y,
        "car/accelerometer_z": accel_z,

        "car/gyrosope_x": gyro_x,
        "car/gyroscope_y": gyro_y,
        "car/gyroscope_z": gyro_z,

        "car/magnetometer_x": magneto_x,
        "car/magnetometer_y": magneto_y,
        "car/magnetometer_z": magneto_z,
    }


def format_detections(yolo_objects, names):
    box_data = []
    # add bboxes
    for det in yolo_objects:
        if len(det):
            for *xyxy, conf, cls in reversed(det):
                c = int(cls)  # integer class
                label = names[c]
                minX, minY, maxX, maxY = xyxy
                box = {
                    "position": {
                        "minX": int(minX),
                        "maxX": int(maxX),
                        "minY": int(minY),
                        "maxY": int(maxY)
                    },
                    "domain": "pixel",
                    "class_id": c,
                    "box_caption": label,
                    "scores": {
                        "conf": float(conf),
                    }
                }

                box_data.append(box)
    boxes = {
        "predictions": {
            "box_data": box_data
        }
    }

    return boxes


def drive(car, camera, mpu, model_trt, yolo_model, config):
    logging.debug("Debug mode enabled")
    logging.info("Starting to drive")

    jetson = jtop()
    jetson.start()

    frame_count = 0

    while True:
        image = camera.read()
        inference_start = time.time()
        debug_log = {}

        imu_values = read_mpu(mpu)
        road_center, objects = infer(image, model_trt, yolo_model)
        car.throttle, car.steering = control_policy(
            road_center,
            objects,
            config
        )

        if config.debug:
            frame_count += 1
            if frame_count % config.debug_freq == 0:
                logging.debug("logging image")
                image = show_label(image, road_center)
                image = cv2.cvtColor(
                    image, cv2.COLOR_BGR2RGB
                )

                boxes = None
                if config.yolo:
                    boxes = format_detections(objects, yolo_model.names)
                    logging.debug(boxes)

                debug_log = {
                    "inference/frame": wandb.Image(image, boxes=boxes),
                }

            is_done = frame_count == config.framerate * config.debug_seconds
            if is_done:
                logging.debug(f"frame count: {frame_count}")
                logging.debug("end debug")
                break

        system_stats = format_jetson_stats(jetson.stats)

        inference_end = time.time()
        inference_seconds = (inference_end - inference_start)

        log = {
            "inference/seconds": inference_seconds,
            "car/steering": car.steering,
            "car/throttle": car.throttle
        }

        wandb.log({**log, **debug_log, **system_stats, **imu_values})


def main(args):
    with wandb.init(
        project=args.project,
        job_type="inference",
        config=args,
        entity=args.entity,
    ) as run:

        config = run.config
        setup_logging(config)

        car, camera, mpu, model_trt, yolo_model = setup(config)

        try:
            drive(car, camera, mpu, model_trt, yolo_model, config)
        except KeyboardInterrupt:
            pass


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run the optimized model on the car",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        "--framerate",
        type=int,
        default=10,
        help="How many images to analyze per second"
    )
    parser.add_argument(
        "--yolo",
        action="store_true",
        help="If specified, will run images through yolo \
             and log predictions to wandb.",
    )
    parser.add_argument(
        "-d",
        "--debug",
        action="store_true"
    )
    parser.add_argument(
        "--debug_seconds",
        type=int,
        default=120,
        help="How long should it run for (in seconds)",
    )
    parser.add_argument(
        "--debug_freq",
        type=int,
        default=10,
        help="How many frames between each logged image",
    )
    parser.add_argument(
        "--project",
        type=str,
        default="racecar",
        help="In which project to log this run"
    )
    parser.add_argument(
        "-e",
        "--entity",
        type=str,
        default=None,
        help="Entity the project belongs to. None = you."
    )
    parser.add_argument(
        "--throttle",
        type=float,
        default=0.005,
        help="Car throttle. Between 0 (full stop) and 1 (full speed).",
    )
    parser.add_argument(
        "-v",
        "--model_version",
        type=str,
        default="latest",
        help="Which artifacts version to use.",
    )
    parser.add_argument(
        "--local_model",
        type=str,
        help="Path to local model. Bypasses artifacts if specified.",
    )
    # TODO add model version as param
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(args)
