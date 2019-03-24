import os
import re
import sys
import logging
from logging.handlers import TimedRotatingFileHandler
from datetime import datetime
from pymeasure.display.Qt import QtGui
from analogreadout.daq import DAQ
from analogreadout.procedures import Pulse2
from mkidplotter import (PulseGUI, PulsePlotWidget, NoisePlotWidget, get_image_icon)
                         
daq = None


def setup_logging():
    log = logging.getLogger()
    directory = os.path.abspath(os.path.join(os.path.dirname(__file__), 'logs'))
    if not os.path.isdir(directory):
        os.mkdir(directory)
    file_name = datetime.now().strftime("pulse_%y%m%d.log")
    file_path = os.path.join(directory, file_name)
    handler = TimedRotatingFileHandler(file_path, when="midnight")
    handler.suffix = "%Y%m%d"
    handler.extMatch = re.compile(r"^\d{8}$")
    handler.setLevel("INFO")
    log_format = logging.Formatter(fmt='%(asctime)s : %(message)s (%(levelname)s)', datefmt='%I:%M:%S %p')
    handler.setFormatter(log_format)
    log.addHandler(handler)

    return log

def pulse_window(other_daq=None):
    x_list = (('t', 't'), ('f1_psd', 'f1_psd'), ('t', 't'), ('f2_psd', 'f2_psd'))
    y_list = (('i1', 'q1'), ('i1_psd', 'q1_psd'), ('i2', 'q2'), ('i2_psd', 'q2_psd'))
    x_label = ("time [µs]", "frequency [Hz]", "time [µs]", "frequency [Hz]")
    y_label = ("Signal [V]", "PSD [V² / Hz]", "Signal [V]", "PSD [V² / Hz]")
    legend_list = (('I', 'Q'), ('I', 'Q'), ('I', 'Q'), ('I', 'Q'))
    widgets_list = (PulsePlotWidget, NoisePlotWidget, PulsePlotWidget, NoisePlotWidget)
    names_list = ('Channel 1: Data', 'Channel 1: Noise', 'Channel 2: Data', 'Channel 2: Noise')
    w = PulseGUI(Pulse2, x_axes=x_list, y_axes=y_list, x_labels=x_label, y_labels=y_label,
                 legend_text=legend_list, plot_widget_classes=widgets_list, plot_names=names_list)
    # connect the daq to the process after making the window so that the log widget gets
    # the instrument creation log messages
    global daq
    if other_daq is not None:
        Pulse2.connect_daq(other_daq)
    elif daq is not None:
        Pulse2.connect_daq(daq)
    else:
        daq = DAQ("UCSB")
        Pulse2.connect_daq(daq)
    return w


if __name__ == '__main__':
    # TODO: implement JPL/UCSB configuration switching
    setup_logging()
    app = QtGui.QApplication(sys.argv)
    # TODO: add pulse image for icon
    # app.setWindowIcon(get_image_icon("pulse.png"))
    window = pulse_window()
    window.activateWindow()
    window.show()
    ex = app.exec_()
    del app  # prevents unwanted segfault on closing the window
    sys.exit(ex)
