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
    default_project = "racecar"
    default_entity = None

    parser = argparse.ArgumentParser(
        description="Upload directory/dataset to wandb"
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
        default=default_project,
        type=str,
        help="Wandb project in which to upload the artifact. "
             f"Default {default_project}",
    )
    parser.add_argument(
        "--entity",
        type=str,
        default=default_entity,
        help=f"Entity the project belongs to. Default {default_entity} (you)"
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(args)
