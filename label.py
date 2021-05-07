import argparse
import wandb
import cv2
import os


# https://www.geeksforgeeks.org/displaying-the-coordinates-of-the-points-clicked-on-the-image-using-python-opencv/
def click_event(event, x, y, flags, path):
    """displays the coordinates of the point clicked on the image."""

    # checking for left mouse clicks
    if event == cv2.EVENT_LBUTTONDOWN:
        # displaying the coordinates
        # on the Shell
        print(x, ' ', y)
        # rename image
        print(path)
        fname = os.path.basename(path)
        directory = os.path.dirname(path)

        fname = "{x}_{y}_{fname}".format(x=x, y=y, fname=fname)

        os.rename(path, os.path.join(directory, fname))
        tmp = img.copy()
        cv2.circle(tmp, (x, y), 5, (0, 255, 0), 2)
        cv2.imshow('image', tmp)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='tool to label images for regression'
    )
    parser.add_argument('--project', type=str, default="racecar")
    parser.add_argument('--entity', type=str, default=None)
    parser.add_argument('dataset', type=str, default=None)

    args = parser.parse_args()

    with wandb.init(
        project=args.project,
        config=args,
        entity=args.entity,
        job_type="labelling"
    ) as run:

        config = run.config

        # TODO check if latest tag needed
        artifact_loc = '{entity}/{proj}/{ds}:latest'.format(
            entity=config.entity,
            proj=config.project,
            ds=config.dataset
        )
        print("downloading ", artifact_loc)
        dataset = run.use_artifact(artifact_loc, type='dataset')
        artifact_dir = dataset.download()

        try:
            for fname in os.listdir(artifact_dir):
                already_labeled = "_" in fname
                if fname.endswith(".jpg") and not already_labeled:
                    # reading the image
                    path = os.path.join(artifact_dir, fname)
                    img = cv2.imread(path, 1)
                    print(fname)

                    # displaying the image
                    cv2.imshow('image', img)

                    # setting mouse handler for the image
                    # and calling the click_event() function
                    cv2.setMouseCallback('image', click_event, path)

                    # wait for any key to be pressed
                    cv2.waitKey(0)

                    # close the window
                    cv2.destroyAllWindows()
        except KeyboardInterrupt:
            pass

        # save changes
        artifact = wandb.Artifact(config.dataset, type="dataset")
        artifact.add_dir(artifact_dir)
        run.log_artifact(artifact)
