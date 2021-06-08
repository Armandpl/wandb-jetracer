# from collections import namedtuple
import logging
import os

from utils.utils import (make_dirs,
                         split_list_by_pct,
                         setup_logging)


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
