import logging
from colorlog import ColoredFormatter

def setup_logger(name: str, log_file: str = "app.log", level: int = logging.INFO, topic: str = "") -> logging.Logger:
    logger = logging.getLogger(name)
    logger.setLevel(level)

    file_handler = logging.FileHandler(log_file)
    console_handler = logging.StreamHandler()

    color_formatter = ColoredFormatter(
        f"%(log_color)s%(asctime)s - {topic} - %(name)s - %(levelname)s - %(message)s",
        log_colors={
            "DEBUG": "cyan",
            "INFO": "green",
            "WARNING": "yellow",
            "ERROR": "red",
            "CRITICAL": "bold_red",
        },
    )

    plain_formatter = logging.Formatter(
        f"%(asctime)s - {topic} - %(name)s - %(levelname)s - %(message)s"
    )

    file_handler.setFormatter(plain_formatter)
    console_handler.setFormatter(color_formatter)

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger

main_logger = setup_logger("semantic-segmentation-trainer", "trainer.log", topic="MAIN")
