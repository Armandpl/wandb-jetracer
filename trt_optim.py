import argparse
import logging
import os

import torch
import torchvision
import wandb

from torch2trt import torch2trt


def main(args):
    with wandb.init(
        project=args.project,
        config=args,
        job_type="trt-optimization",
        entity=args.entity
    ) as run:

        logging.info("downloading non optimized model")
        artifact = run.use_artifact("model:latest")
        artifact_dir = artifact.download()

        logging.info("creating model architecture")
        # fetching the model architecture from the producer run
        producer_run = artifact.logged_by()
        model = torchvision.models.__dict__[
            producer_run.config["architecture"]
        ](pretrained=False)

        model.fc = torch.nn.Linear(model.fc.in_features, 2)
        model = model.cuda().eval().half()
        model.load_state_dict(
            torch.load(os.path.join(artifact_dir, "model.pth"))
        )

        # dummy input
        data = torch.zeros((1, 3, 224, 224)).cuda().half()

        print("optimizing model...")
        model_trt = torch2trt(model, [data], fp16_mode=True)

        # evaluate model
        # TODO

        # save the model for inference
        print("saving model and uploading it to wandb...")
        torch.save(model_trt.state_dict(), "trt-model.pth")

        trt_artifact = wandb.Artifact("trt-model", type="model")
        trt_artifact.add_file("trt-model.pth")
        run.log_artifact(trt_artifact)


def parse_args():
    default_entity = None
    default_project = "racecar"

    parser = argparse.ArgumentParser(
        description="Pull the latest trained model, optimize it and log it."
    )
    parser.add_argument(
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
