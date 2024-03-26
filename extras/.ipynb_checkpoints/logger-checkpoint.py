import logging
import sys

class LoggerHandler(logging.Handler):
    """ LoggerHandler """
    def __init__(self):
        """ __init__ """
        super().__init__()
        self.log = ""

    def reset(self):
        """ reset """
        self.log = ""

    def emit(self, record):
        """ emit """
        if record.name == "httpx":
            return
        log_entry = self.format(record)
        self.log += log_entry
        self.log += "\n\n"

def get_logger(name: str) -> logging.Logger:
    """ Gets a standard logger with a stream hander to stdout. """
    formatter = logging.Formatter(fmt="%(asctime)s - %(levelname)s - %(name)s - %(message)s", datefmt="%m/%d/%Y %H:%M:%S")
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(formatter)
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.addHandler(handler)
    return logger

def reset_logging() -> None:
    """ reset_logging """
    root = logging.getLogger()
    list(map(root.removeHandler, root.handlers))
    list(map(root.removeFilter, root.filters))