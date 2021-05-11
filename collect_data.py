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
    # TODO: find where these number belong?
    default_nb_imgs = 120
    default_framerate = 2
    default_img_size = 224
    default_entity = None
    default_project = "racecar"

    parser = argparse.ArgumentParser(
        description="Collect square images and upload them to wandb."
    )
    parser.add_argument(
        "dataset_name",
        type=str,
        help="The name of the dataset artifact to add to."
    )
    parser.add_argument(
        "-n",
        "--nb_imgs",
        type=int,
        default=default_nb_imgs,
        help=f"Number of images to collect. Default {default_nb_imgs}",
    )
    parser.add_argument(
        "--framerate",
        type=int,
        default=default_framerate,
        help=f"Number of images to collect/s. Default {default_framerate}"
    )
    parser.add_argument(
        "--img_size",
        type=int,
        default=default_img_size,
        help=f"Size of the images to collect. Default {default_img_size}"
    )
    parser.add_argument(
        "-e",
        "--entity",
        type=str,
        default=default_entity,
        help=f"Which entity owns the project. Default {default_entity} (you)"
    )
    parser.add_argument(
        "--project",
        type=str,
        default=default_project,
        help=f"Project the dataset belongs to. Default {default_project}"
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(args)
