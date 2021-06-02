import logging
import os


def make_dirs(output_dataset):
    out_dirs = [os.path.join(output_dataset, split)
                for split in ["train", "val", "test"]]

    [os.makedirs(curr_dir) for curr_dir in out_dirs]

    return out_dirs


def split_list_by_pct(data, pcts):
    sizes = [int(pct*len(data)) for pct in pcts]

    it = iter(data)
    return [[next(it) for _ in range(size)] for size in sizes]


def setup_logging(config=None):
    if config is None:
        logging_level = logging.INFO
    else:
        logging_level = logging.DEBUG if config.debug else logging.INFO

    logging.basicConfig(
        format='%(levelname)s: %(message)s',
        level=logging_level
    )
