import argparse
import time

import cv2
import torch
import wandb

from jetcam.csi_camera import CSICamera
from jetracer.nvidia_racecar import NvidiaRacecar
from utils.utils import preprocess
from torch2trt import TRTModule

THROTTLE_GAIN = -1
STEERING_GAIN = -1


def setup(config):

    print("downloading latest optimized model")
    # artifact = run.use_artifact('trt-model:latest')
    # artifact_dir = artifact.download()

    print("loading state dict")
    model_trt = TRTModule()
    # model_trt.load_state_dict(torch.load(
    #     os.path.join(artifact_dir, 'trt-model.pth')
    # ))
    # model_trt.load_state_dict(torch.load('trt-model.pth'))
    model_trt.load_state_dict(
        torch.load("../jetracer/notebooks/road_following_model_trt.pth")
    )

    print("setting up car and camera")
    car = NvidiaRacecar()
    camera = CSICamera(
        width=224, height=224, capture_fps=config.framerate
    )

    return car, camera, model_trt


def drive(car, camera, model_trt, config):
    if config.debug:
        print("Debug mode enabled")
        frame_count = 0
    print("Starting to drive")
    ii = 0
    jj = 0
    car.throttle = config.throttle*THROTTLE_GAIN
    while True:
        image = camera.read()
        start = time.time()
        if config.debug:
            unprocessed = image.copy()

        image = preprocess(image).half()
        output = model_trt(image).detach().cpu().numpy().flatten()
        x = float(output[0])
        y = float(output[1])
        car.steering = STEERING_GAIN * x  # *(y+1)/2

        wandb.log(
            {
                "car_log_idx": jj,
                "car/steering": car.steering,
                "car/throttle": car.throttle,
            }
        )
        jj += 1

        if config.debug:
            frame_count += 1
            if frame_count % config.debug_freq == 0:
                print("logging image")
                x = int((x + 1) / 2 * 224)
                y = int((y + 1) / 2 * 224)
                cv2.circle(unprocessed, (x, y), 5, (0, 255, 0), 2)
                unprocessed = cv2.cvtColor(
                    unprocessed, cv2.COLOR_BGR2RGB
                )
                wandb.log(
                    {
                        "frame_idx": ii,
                        "inference/frame": wandb.Image(unprocessed),
                    }
                )
                ii += 1

            if (
                frame_count
                == config.framerate * config.debug_seconds
            ):
                print("frame count: ", frame_count)
                print("end debug")
                break
        end = time.time()
        wandb.log({"inference/seconds": (end - start)})


def main(args):
    with wandb.init(
        project=args.project,
        job_type="inference",
        config=args,
        entity=args.entity,
    ) as run:

        config = run.config

        car, camera, model_trt = setup(config)

        try:
            drive(car, camera, model_trt, config)
        except KeyboardInterrupt:
            pass


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run the optimized model on the car"
    )

    parser.add_argument("--framerate", type=int, default=10)
    parser.add_argument("--debug", type=bool, default=False)
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
        help="how many frame between each logged image",
    )
    parser.add_argument("--project", type=str, default="racecar")
    parser.add_argument("--entity", type=str, default=None)
    parser.add_argument(
        "--throttle",
        type=float,
        default=0.005,
        help="Car throttle. Between 0 (full stop) and 1 (full speed).",
    )
    # TODO add policy as a param
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(args)
