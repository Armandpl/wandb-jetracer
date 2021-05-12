import argparse
import logging
import os
import time

import cv2
import torch
import wandb

from jetcam.csi_camera import CSICamera
from jetracer.nvidia_racecar import NvidiaRacecar
from utils.xy_dataset import preprocess
from utils.utils import setup_logging
from torch2trt import TRTModule

THROTTLE_GAIN = -1
STEERING_GAIN = -1
IMG_SIZE = 224


def setup(config):

    model_trt = TRTModule()

    if config.local_model is None:
        logging.info("Downloading latest optimized model...")
        artifact = wandb.use_artifact('trt-model:latest')
        artifact_dir = artifact.download()
        model_trt.load_state_dict(torch.load(
            os.path.join(artifact_dir, 'trt-model.pth')
        ))
    else:
        logging.info(f"Using local model: {config.local_model}")
        model_trt.load_state_dict(torch.load(config.local_model))

    logging.info("Setting up car and camera")
    car = NvidiaRacecar()
    camera = CSICamera(
        width=IMG_SIZE, height=IMG_SIZE, capture_fps=config.framerate
    )

    return car, camera, model_trt


def control_policy(road_center, config):
    x, y = road_center
    steering = x * STEERING_GAIN  # *(y+1)/2
    throttle = config.throttle * THROTTLE_GAIN

    return throttle, steering


def infer(image, model_trt):
    image = preprocess(image).half()
    output = model_trt(image).squeeze()  # .detach().cpu().numpy().flatten()
    x, y = float(output[0]), float(output[1])

    return x, y


def show_label(image, coordinates):
    x, y = coordinates

    x = int((x + 1) / 2 * 224)
    y = int((y + 1) / 2 * 224)
    cv2.circle(image, (x, y), 5, (0, 255, 0), 2)
    image = cv2.cvtColor(
        image, cv2.COLOR_BGR2RGB
    )

    return image


def drive(car, camera, model_trt, config):
    logging.debug("Debug mode enabled")
    logging.info("Starting to drive")

    frame_count = 0

    while True:
        image = camera.read()
        inference_start = time.time()
        debug_log = {}

        if config.debug:
            unprocessed = image.copy()

        x, y = infer(image, model_trt)
        car.throttle, car.steering = control_policy((x, y), config)

        if config.debug:
            frame_count += 1
            if frame_count % config.debug_freq == 0:
                logging.debug("logging image")
                unprocessed = show_label(unprocessed, (x, y))
                debug_log = {
                    "inference/frame": wandb.Image(unprocessed),
                }

            is_done = frame_count == config.framerate * config.debug_seconds
            if is_done:
                logging.debug(f"frame count: {frame_count}")
                logging.debug("end debug")
                break

        inference_end = time.time()
        inference_seconds = (inference_end - inference_start)

        log = {
            "inference/seconds": inference_seconds,
            "car/steering": car.steering,
            "car/throttle": car.throttle,
        }

        wandb.log({**log, **debug_log})


def main(args):
    with wandb.init(
        project=args.project,
        job_type="inference",
        config=args,
        entity=args.entity,
    ) as run:

        config = run.config
        setup_logging(config)

        car, camera, model_trt = setup(config)

        try:
            drive(car, camera, model_trt, config)
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
        "-d",
        "--debug",
        action="store_true"
    )
    parser.add_argument(
        "--debug_seconds",
        type=int,
        default=120,
        help="how long should it run for (in seconds)",
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
        "--local_model",
        type=str,
        help="Path to local model. If specified, bypasses artifacts.",
    )
    # TODO add policy as a param
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(args)
