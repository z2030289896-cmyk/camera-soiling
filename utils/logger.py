"""统一日志工具。创建名为 camera_soiling 的 logger，输出到 stdout。"""
import logging


def get_logger():

    logger = logging.getLogger("camera_soiling")

    logger.setLevel(logging.INFO)

    if not logger.handlers:

        handler = logging.StreamHandler()

        formatter = logging.Formatter(
            "[%(asctime)s] %(message)s"
        )

        handler.setFormatter(formatter)

        logger.addHandler(handler)

    return logger