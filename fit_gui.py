import os
import sys
import yaml
import logging
from datetime import datetime
from pymeasure.display.Qt import QtGui
from mkidplotter import FitGUI

log = logging.getLogger(__name__)
log.addHandler(logging.NullHandler())


def get_config(file_name):
    file_path = os.path.join(os.getcwd(), file_name)
    if not os.path.isfile(file_path):  # check configurations folder
        file_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                 'analogreadout', 'configurations', file_name.lower() + ".yaml")
    with open(file_path, "r") as f:
        cfg = yaml.load(f, Loader=yaml.Loader)
    return cfg


def setup_logging():
    logger = logging.getLogger()
    directory = os.path.abspath(os.path.join(os.path.dirname(__file__), 'logs'))
    if not os.path.isdir(directory):
        os.mkdir(directory)
    name = datetime.now().strftime("analogreadout_%y%m%d.log")
    path = os.path.join(directory, name)
    handler = logging.FileHandler(path, mode='a', encoding='utf-8')
    handler.setLevel("INFO")
    log_format = logging.Formatter(
        datefmt='%m/%d/%Y %I:%M:%S %p',
        fmt='%(asctime)s : %(message)s (%(levelname)s)')
    handler.setFormatter(log_format)
    logger.addHandler(handler)
    # Hide very verbose libraries
    logging.getLogger("lakeshore").setLevel(logging.WARNING)
    logging.getLogger("matplotlib").setLevel(logging.WARNING)
    return logger


def fit_window(configuration):
    # make the window
    w = FitGUI(**configuration['gui']['fit'])
    return w


if __name__ == '__main__':
    # Set up the logging.
    setup_logging()

    # Open the configuration file.
    config = get_config(sys.argv.pop(1) if len(sys.argv) > 1 else "ucsb")

    # Create the window.
    try:
        app = QtGui.QApplication(sys.argv)
        # app.setWindowIcon(get_image_icon("fit.png"))
        window = fit_window(config)
        window.activateWindow()
        window.show()
        ex = app.exec_()
        del app  # prevents unwanted segfault on closing the window
        sys.exit(ex)
    except Exception as error:
        log.exception(error)
        raise error
