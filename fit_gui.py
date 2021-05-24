import os
import sys
import yaml
import logging
from datetime import datetime
from pymeasure.display.Qt import QtGui
from mkidplotter import FitGUI


def setup_logging():
    log = logging.getLogger()
    directory = os.path.abspath(os.path.join(os.path.dirname(__file__), 'logs'))
    if not os.path.isdir(directory):
        os.mkdir(directory)
    file_name = datetime.now().strftime("analogreadout_%y%m%d.log")
    file_path = os.path.join(directory, file_name)
    handler = logging.FileHandler(file_path, mode='a')
    handler.setLevel("INFO")
    log_format = logging.Formatter(fmt='%(asctime)s : %(message)s (%(levelname)s)', datefmt='%m/%d/%Y %I:%M:%S %p')
    handler.setFormatter(log_format)
    log.addHandler(handler)
    return log


def fit_window(configuration):
    # make the window
    w = FitGUI(**configuration['gui']['fit'])
    return w


if __name__ == '__main__':
    # Set up the logging.
    setup_logging()

    # Open the configuration file.
    file_name = sys.argv.pop(1) if len(sys.argv) > 1 else "ucsb"
    if not os.path.isfile(file_name):  # check configurations folder
        file_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                 'analogreadout', 'configurations', file_name.lower() + ".yaml")
    with open(file_path, "r") as f:
        config = yaml.load(f, Loader=yaml.Loader)

    # Create the window.
    app = QtGui.QApplication(sys.argv)
    # app.setWindowIcon(get_image_icon("fit.png"))
    window = fit_window(config)
    window.activateWindow()
    window.show()
    ex = app.exec_()
    del app  # prevents unwanted segfault on closing the window
    sys.exit(ex)
