import json
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


def log_event(
        logger,
        event: str,
        **fields,
):
    """
    Write one structured observability event.

    The outer logging format remains human-readable,
    while event fields are serialized as JSON so they
    can also be parsed by log-processing systems.
    """

    payload = {
        key: value
        for key, value in fields.items()
        if value is not None
    }

    logger.info(
        "%s | %s",
        event,
        json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
            default=str,
        ),
    )