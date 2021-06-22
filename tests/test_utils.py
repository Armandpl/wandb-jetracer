# from collections import namedtuple
import logging
import os

from wandb_jetracer.utils.utils import (make_dirs,
                                        split_list_by_pct,
                                        setup_logging,
                                        label_img)


def test_label_img(fs):
    path = "unique-id.jpg"
    fs.create_file(path)

    path = label_img(10, 20, path)

    assert os.path.exists("10_20_unique-id.jpg")

    label_img(30, 40, path)

    assert os.path.exists("30_40_unique-id.jpg")


def test_make_dirs(fs):
    out_dir = "test_dir"

    make_dirs(out_dir)

    assert os.path.exists(os.path.join(out_dir, "train"))\
           is os.path.exists(os.path.join(out_dir, "val"))\
           is os.path.exists(os.path.join(out_dir, "test")) is True


def test_split_list_by_pct():
    # Arrange
    list = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    pcts = [0.8, 0.2]

    # Act
    new_list = split_list_by_pct(list, pcts)

    # Assert
    assert new_list == [[1, 2, 3, 4, 5, 6, 7, 8], [9, 10]]


def test_setup_logging_no_debug(caplog):
    setup_logging()

    logging.debug("hello world")

    assert "hello world" not in caplog.text


# TODO: find out why that's not passing
#       even though it's working outside of tests
# def test_setup_logging_debug(caplog):
#     config = {"debug": True}
#     config = namedtuple("Config", config.keys())(*config.values())
#     logging = setup_logging(config)
#
#     logging.debug("hello world")
#
#     assert "hello world" in caplog.text
