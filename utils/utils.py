import logging


def setup_logging(config=None):
    if config is None:
        logging_level = logging.INFO
    else:
        logging_level = logging.DEBUG if config.debug else logging.INFO

    logging.basicConfig(
        format='%(levelname)s: %(message)s',
        level=logging_level
    )
