import logging
from pathlib import Path
from datetime import datetime
import time


class Log():
    def __init__(self, path):
        # logging set up
        Path.mkdir(path, parents=True, exist_ok=True)

        self.rootLogger = logging.getLogger()
        self.rootLogger.setLevel(logging.DEBUG)
        logFormatter = logging.Formatter(
            "%(asctime)s - %(levelname)s - %(message)s")

        self.fileHandler = logging.FileHandler(path.joinpath(
            f'{datetime.today().strftime("%m-%d--%H-%M-%S")}.log'))
        self.fileHandler.setFormatter(logFormatter)
        self.rootLogger.addHandler(self.fileHandler)

        self.consoleHandler = logging.StreamHandler()
        self.consoleHandler.setFormatter(logFormatter)
        self.rootLogger.addHandler(self.consoleHandler)
        self.start_time_entire = time.time()

    def detach(self):
        running_time = time.time() - self.start_time_entire
        self.info(
            f"Total running time: {int(running_time//60)}:{int(running_time%60):0>2d}\n")
        self.rootLogger.removeHandler(self.fileHandler)
        self.rootLogger.removeHandler(self.consoleHandler)

    def info(self, mess):
        logging.info(mess)
