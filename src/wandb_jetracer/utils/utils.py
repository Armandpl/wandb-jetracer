import logging
import os
import uuid

import cv2


def make_dirs(output_dataset):
    out_dirs = [os.path.join(output_dataset, split)
                for split in ["train", "val", "test"]]

    [os.makedirs(curr_dir) for curr_dir in out_dirs]

    return out_dirs


def split_list_by_pct(data, pcts):
    sizes = [int(pct*len(data)) for pct in pcts]

    it = iter(data)
    return [[next(it) for _ in range(size)] for size in sizes]


def create_img_name():
    return str(uuid.uuid1()) + ".jpg"


def label_img(x, y, path):
    """ save x, y coordinates in filename"""

    fname = os.path.basename(path)
    directory = os.path.dirname(path)

    fname = fname.split("_")[-1]
    fname = f"{x}_{y}_{fname}"
    new_path = os.path.join(directory, fname)
    os.rename(path, new_path)
    logging.debug(f"Rename {path} to {new_path}")

    return new_path


def setup_logging(config=None):
    if config is None:
        logging_level = logging.INFO
    else:
        logging_level = logging.DEBUG if config.debug else logging.INFO

    logging.basicConfig(
        format='%(levelname)s: %(message)s',
        level=logging_level
    )


def show_label(image, coordinates, color=(0, 255, 0)):
    """
    Show a circle at x, y coordinates on image
    x, y belong to [-1, 1]
    """
    img_h, img_w, _ = image.shape
    x, y = coordinates

    # shift x, y to be between 0 and 1
    x = int((x + 1) / 2 * img_w)
    y = int((y + 1) / 2 * img_h)

    cv2.circle(image, (x, y), 5, color, 2)

    return image


def torch2cv2(tensor):
    img = tensor.permute(1, 2, 0).cpu().numpy()*255
    img = cv2.cvtColor(
                img, cv2.COLOR_BGR2RGB
                )

    return img
