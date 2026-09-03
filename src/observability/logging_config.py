import logging


LOGGER_NAME = "ecommerce_rag"


def configure_logging(
    level="INFO",
):

    logger = logging.getLogger(
        LOGGER_NAME
    )

    log_level = getattr(
        logging,
        str(level).upper(),
        logging.INFO,
    )

    logger.setLevel(
        log_level
    )

    if not logger.handlers:

        handler = logging.StreamHandler()

        formatter = logging.Formatter(
            "%(asctime)s | "
            "%(levelname)s | "
            "%(name)s | "
            "%(message)s"
        )

        handler.setFormatter(
            formatter
        )

        logger.addHandler(
            handler
        )

    logger.propagate = False

    return logger


def get_logger(
    name,
):

    return logging.getLogger(
        f"{LOGGER_NAME}.{name}"
    )