import argparse
import logging
import os

import cv2
import wandb

from utils.utils import setup_logging


class ImageLabeller:
    def __init__(self, directory):
        self.directory = directory
        self.count_deleted = 0

        self.get_non_labelled_imgs()
        logging.info(f"{len(self.to_label)} images to label out of "
                     f"{len(os.listdir(self.directory))}")

        to_label_len = len(self.to_label)

        self.label()

        self.get_non_labelled_imgs()
        self.count_labelled = to_label_len - len(self.to_label)
        self.count_labelled -= self.count_deleted

        logging.info(f"{self.count_labelled} images labelled, "
                     f"{self.count_deleted} deleted.")

    def get_non_labelled_imgs(self):
        self.to_label = []
        for fname in os.listdir(self.directory):
            already_labeled = "_" in fname
            if fname.endswith(".jpg") and not already_labeled:
                self.to_label.append(fname)

    def label(self):
        for fname in self.to_label:
            # reading the image
            self.current_path = os.path.join(self.directory, fname)
            self.current_img = cv2.imread(self.current_path, 1)

            # displaying the image
            imshow_fullscreen(self.current_img)

            # setting mouse handler for the image
            # and calling the click_event() function
            cv2.setMouseCallback("image", self.click_event)

            # wait for key to be pressed
            key = cv2.waitKey(0)
            if key == 100:  # d
                os.remove(self.current_path)
                self.count_deleted += 1
                logging.info(f"deleted {self.current_path}")
            elif key == 113:  # q
                cv2.destroyAllWindows()
                break

            cv2.destroyAllWindows()

    def click_event(self, event, x, y, flags, param):
        """Display the clicked coordinates and add those the image filename"""

        # checking for left mouse clicks
        if event == cv2.EVENT_LBUTTONDOWN:
            # displaying the coordinates
            logging.debug(f"{x}, {y} clicked")
            self.current_path = label_img(x, y, self.current_path)

            tmp = self.current_img.copy()
            cv2.circle(tmp, (x, y), 5, (0, 255, 0), 2)
            cv2.imshow("image", tmp)


def imshow_fullscreen(img):
    cv2.namedWindow("image", cv2.WND_PROP_FULLSCREEN)
    cv2.setWindowProperty(
        "image",
        cv2.WND_PROP_FULLSCREEN,
        cv2.WINDOW_FULLSCREEN
    )
    cv2.imshow("image", img)


def label_img(x, y, path):
    # save coordinates in filename logging("labelled {path}")
    fname = os.path.basename(path)
    directory = os.path.dirname(path)

    fname = fname.split("_")[-1]
    fname = f"{x}_{y}_{fname}"
    new_path = os.path.join(directory, fname)
    os.rename(path, new_path)
    logging.debug(f"Rename {path} to {new_path}")

    return new_path


def main(args):
    with wandb.init(
        project=args.project,
        config=args,
        entity=args.entity,
        job_type="labelling",
    ) as run:

        config = run.config

        setup_logging(config)

        # download dataset
        artifact_loc = f"{config.entity}/{config.project}/{config.dataset}:latest"
        logging.info(f"downloading {artifact_loc}")
        dataset = run.use_artifact(artifact_loc, type="dataset")
        artifact_dir = dataset.download()

        labeller = ImageLabeller(artifact_dir)

        if labeller.count_labelled > 0:
            logging.info("Saving changes to artifact")
            artifact = wandb.Artifact(config.dataset, type="dataset")
            artifact.add_dir(artifact_dir)
            run.log_artifact(artifact)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Tool to label images for regression. "
                    "Save x,y coords in the filename. \n"
                    "Controls: Label: click, delete img: d, quit: q, "
                    "next image: any key",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("dataset", type=str, help="Dataset artifact to label")
    parser.add_argument(
        "--project",
        type=str,
        default="racecar",
        help="Project the dataset belongs to."
    )
    parser.add_argument(
        "-e",
        "--entity",
        type=str,
        default=None,
        help="Entity the project belongs to. None = you."
    )
    parser.add_argument(
        "-d",
        "--debug",
        action="store_true",
        help="Flag to display debug messages."
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(args)
