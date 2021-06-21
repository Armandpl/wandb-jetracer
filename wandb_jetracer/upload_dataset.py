import argparse
import wandb


def main(args):
    with wandb.init(
        project=args.project,
        config=args,
        job_type="upload-dataset",
        entity=args.entity
    ) as run:

        config = run.config

        artifact = wandb.Artifact(
            config.name,
            type="dataset",
            description="Images of the racetrack from the car pov."
                        "With the center of the track annotated.",
            metadata={
                "label-format": "Center of the track in pixels."
                "Stored in the filename: 'x_y_uid.png'"
            },
        )

        artifact.add_dir(config.directory)

        run.log_artifact(artifact)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Upload directory/dataset to wandb",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "directory",
        type=str,
        help="Directory containing the dataset to upload"
    )
    parser.add_argument(
        "name",
        type=str,
        help="Dataset name."
    )
    parser.add_argument(
        "--project",
        default="racecar",
        type=str,
        help="Wandb project in which to upload the artifact."
    )
    parser.add_argument(
        "-e",
        "--entity",
        type=str,
        default=None,
        help="Entity the project belongs to. None = you"
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(args)
