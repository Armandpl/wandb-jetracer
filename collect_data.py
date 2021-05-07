import os
import cv2
import uuid
import wandb
import argparse
from tqdm import tqdm
from jetcam.csi_camera import CSICamera

if __name__ == "__main__":
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
    parser.add_argument("--out_dir", type=str, default="./data")
    parser.add_argument("--entity", type=str, default=None)
    parser.add_argument("--project", type=str, default="racecar")

    args = parser.parse_args()

    with wandb.init(
        project=args.project,
        config=args,
        job_type="collect-data",
        entity=args.entity,
    ) as run:

        config = run.config

        # first pull the latest version of the dataset to create the new one
        os.makedirs(config.out_dir, exist_ok=True)

        print("Setting up camera...")
        camera = CSICamera(
            width=config.img_size,
            height=config.img_size,
            capture_fps=config.framerate,
        )

        count_imgs = 0

        dataset = wandb.Artifact(config.dataset_name, type="dataset")

        print("Collecting images...")
        for _ in tqdm(range(config.nb_imgs)):
            image = camera.read()
            fname = str(uuid.uuid1()) + ".jpg"
            cv2.imwrite(os.path.join(config.out_dir, fname), image)

        # add images to artifact
        dataset.add_dir(config.out_dir)
        # log artifact to wandb
        run.log_artifact(dataset)
