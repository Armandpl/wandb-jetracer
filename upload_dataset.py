import argparse
import wandb
import os

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Upload directory/dataset to wandb") 
    parser.add_argument("directory", type=str, help="directory containing the dataset to upload")
    parser.add_argument("--project", default="wandb-jetracer", type=str, help="wandb project in which to upload the artifact")
    parser.add_argument("--entity", type=str, default=None)
    args = parser.parse_args()
    
    with wandb.init(project=args.project, job_type='upload-dataset', entity=args.entity) as run:

        artifact = wandb.Artifact(
            'racetrack', type='dataset',
            description="Images of the racetrack from the car pov. With the center of the track annotated.",
            metadata={
                "label-format":"Center of the track in pixels. Stored in the filename: 'x_y_uid.png'"
            }
        )

        artifact.add_dir(args.directory)

        run.log_artifact(artifact)

    
