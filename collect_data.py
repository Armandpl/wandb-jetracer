import argparse
import logging
import os
import uuid

import cv2
from tqdm import tqdm
import wandb

from jetcam.csi_camera import CSICamera


def collect_images(camera, output_dir, config):
    logging.info("Collecting images...")

    for _ in tqdm(range(config.nb_imgs)):
        image = camera.read()
        fname = create_img_name()
        cv2.imwrite(os.path.join(output_dir, fname), image)


def setup(config):
    try:
        # first pull the latest version of the dataset to add to it
        artifact = wandb.use_artifact(f"{config.dataset_name}:latest")
        artifact_dir = artifact.download()
        logging.info("Dataset already exists, adding to it")
    except wandb.errors.CommError:
        logging.info("Dataset doesn't exist yet, creating it")
        artifact_dir = config.dataset_name
        os.makedirs(artifact_dir, exist_ok=False)

    logging.info("Setting up camera...")

    camera = CSICamera(
        width=config.img_size,
        height=config.img_size,
        capture_fps=config.framerate,
    )
    return camera, artifact_dir


def create_img_name():
    return str(uuid.uuid1()) + ".jpg"


def main(args):
    with wandb.init(
        project=args.project,
        config=args,
        job_type="collect-data",
        entity=args.entity,
    ) as run:

        config = run.config

        camera, output_dir = setup(config)

        collect_images(camera, output_dir, config)

        dataset = wandb.Artifact(config.dataset_name, type="dataset")

        # add images to artifact
        dataset.add_dir(output_dir)

        # log artifact to wandb
        run.log_artifact(dataset)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Collect images and upload them to wandb"
    )
    parser.add_argument("dataset_name", type=str)
    parser.add_argument(
        "--nb_imgs",
        type=int,
        default=120,
        help="how many images you want to collect",
    )
    parser.add_argument("--framerate", type=int, default=2)
    parser.add_argument("--img_size", type=int, default=224)
    parser.add_argument("--entity", type=str, default=None)
    parser.add_argument("--project", type=str, default="racecar")

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(args)
