import argparse
import logging
import os
import sys

import cv2
from tqdm import tqdm
import wandb

from jetcam.csi_camera import CSICamera
from wandb_jetracer.utils.utils import setup_logging, create_img_name


def collect_images(camera, output_dir, config):
    logging.info("Collecting images...")

    for _ in tqdm(range(config.nb_imgs)):
        image = camera.read()
        fname = create_img_name()
        cv2.imwrite(os.path.join(output_dir, fname), image)


def main(args):
    with wandb.init(
        project=args.project,
        config=args,
        job_type="collect-data",
        entity=args.entity,
    ) as run:

        config = run.config
        setup_logging()

        camera, output_dir = setup(config)
        collect_images(camera, output_dir, config)

        dataset = wandb.Artifact(config.dataset_name, type="dataset")
        # add images to artifact
        dataset.add_dir(output_dir)
        # log artifact to wandb
        run.log_artifact(dataset)


def setup(config):
    try:
        # first pull the latest version of the dataset to add to it
        artifact = wandb.use_artifact(f"{config.dataset_name}:latest")
        artifact_dir = artifact.download()
        logging.info("Dataset already exists, adding to it")
    except wandb.errors.CommError:
        logging.info("Dataset doesn't exist yet, creating it")
        artifact_dir = config.dataset_name
        try:
            os.makedirs(artifact_dir, exist_ok=False)
        except FileExistsError:
            logging.error(f"Local directory {artifact_dir} shouldn't exist. "
                          "Please delete/rename it.")
            sys.exit(0)

    logging.info("Setting up camera...")

    camera = CSICamera(
        width=config.img_size,
        height=config.img_size,
        capture_fps=config.framerate,
    )
    return camera, artifact_dir


def parse_args():
    parser = argparse.ArgumentParser(
        description="Collect square images and upload them to wandb.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
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
        default=120,
        help="Total number of images to collect.",
    )
    parser.add_argument(
        "--framerate",
        type=int,
        default=2,
        help="Number of images to collect/s."
    )
    parser.add_argument(
        "--img_size",
        type=int,
        default=224,
        help="Size of the images to collect."
    )
    parser.add_argument(
        "-e",
        "--entity",
        type=str,
        default=None,
        help="Which entity owns the project. None = you"
    )
    parser.add_argument(
        "--project",
        type=str,
        default="racecar",
        help="Project the dataset belongs to."
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(args)
