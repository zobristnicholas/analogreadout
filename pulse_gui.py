#! /usr/bin/env python

import os
import re
import sys
import numpy as np
from datetime import datetime
from collections import deque
from threading import Thread, Event
from pymeasure.display.Qt import QtGui
import logging
from logging.handlers import TimedRotatingFileHandler
from analogreadout.daq import DAQ
from analogreadout.procedures import Pulse2, Pulse1
from mkidplotter import (PulseGUI, SweepPlotWidget, PulsePlotWidget, NoisePlotWidget, ScatterPlotWidget,
                         HistogramPlotWidget, TimePlotIndicator, get_image_icon)
                         
daq = None
temperature_updater = None

time_stamps = deque(maxlen=int(24 * 60))  # one day of data if refresh time is every minute
temperatures = deque(maxlen=int(24 * 60))
refresh_time = 60  # refresh temperature every minute


class Updater(Thread):
    def __init__(self):
        Thread.__init__(self, daemon=True)  # stop process on program exit
        self.finished = Event()
        self.start()

    def cancel(self):
        self.finished.set()

    def run(self):
        while not self.finished.wait(refresh_time):
            self.update()

    @staticmethod
    def update():
        try:
            temp = daq.thermometer.temperature * 1000
            res = daq.thermometer.resistance
            if not np.isnan(temp) and not np.isnan(res):
                time_stamps.append(datetime.now().timestamp())
                temperatures.append(temp)
        except AttributeError:
            pass


def setup_logging():
    log = logging.getLogger()
    directory = os.path.abspath(os.path.join(os.path.dirname(__file__), 'logs'))
    if not os.path.isdir(directory):
        os.mkdir(directory)
    file_name = datetime.now().strftime("analogreadout_%y%m%d.log")
    file_path = os.path.join(directory, file_name)
    handler = TimedRotatingFileHandler(file_path, when="midnight")  # TODO: logfile not rotating properly
    handler.suffix = "%Y%m%d"
    handler.extMatch = re.compile(r"^\d{8}$")
    handler.setLevel("INFO")
    log_format = logging.Formatter(fmt='%(asctime)s : %(message)s (%(levelname)s)', datefmt='%I:%M:%S %p')
    handler.setFormatter(log_format)
    log.addHandler(handler)

    return log

    
def pulse_window(config="UCSB"):
    # setup options
    if config == "UCSB":
        x_list = (('i1_loop', 'i1'), ('f1_psd', 'f1_psd'), ('i2_loop', 'i2'), ('f2_psd', 'f2_psd'), ('peaks1',))
        y_list = (('q1_loop', 'q1'), ('i1_psd', 'q1_psd'), ('q2_loop', 'q2'), ('i2_psd', 'q2_psd'), ('peaks2',))
        x_label = ("I [V]", "frequency [Hz]", "I [V]", "frequency [Hz]", "channel 1 pulse amplitudes |I + iQ|")
        y_label = ("Q [V]", "PSD [V² / Hz]", "Q [V]", "PSD [V² / Hz]", "channel 2 pulse amplitudes |I + iQ|")
        legend_list = (('loop', 'data'), ('I', 'Q'), ('loop', 'data'), ('I', 'Q'), None)
        widgets_list = (PulsePlotWidget, NoisePlotWidget, PulsePlotWidget, NoisePlotWidget, ScatterPlotWidget)
        names_list = ('Channel 1: Data', 'Channel 1: Noise', 'Channel 2: Data', 'Channel 2: Noise', 'Amplitude Scatter')
        procedure_class = Pulse2
    elif config == "JPL":
        x_list = (('i_loop', 'i'), ('f_psd', 'f_psd'), ('hist_x',))
        y_list = (('q_loop', 'q'), ('i_psd', 'q_psd'), ('hist_y',))
        x_label = ("I [V]", "frequency [Hz]", "pulse amplitudes |I + iQ|")
        y_label = ("Q [V]", "PSD [V² / Hz]", "probability density")
        legend_list = (('loop', 'data'), ('I', 'Q'), None)
        widgets_list = (PulsePlotWidget, NoisePlotWidget, HistogramPlotWidget)
        names_list = ('Data', 'Noise', 'Histogram')
        procedure_class = Pulse1
    else:
        raise ValueError("'{}' is not a valid configuration".format(config))
    # setup indicators
    indicators = TimePlotIndicator(time_stamps, temperatures, title='Device Temperature [mK]')
    global temperature_updater
    temperature_updater = Updater()
    # make the window
    w = PulseGUI(procedure_class, x_axes=x_list, y_axes=y_list, x_labels=x_label, y_labels=y_label,
                 legend_text=legend_list, plot_widget_classes=widgets_list, plot_names=names_list,
                 persistent_indicators=indicators)
    # create and connect the daq to the process after making the window so that the log widget gets
    # the instrument creation log messages
    global daq
    if daq is not None:
        procedure_class.connect_daq(daq)
    else:
        daq = DAQ(config)
        procedure_class.connect_daq(daq)
    return w


if __name__ == '__main__':
    setup_logging()
    if len(sys.argv) > 1:
        cfg = sys.argv.pop(1)
    else:
        cfg = "UCSB"
    app = QtGui.QApplication(sys.argv)
    # TODO: add pulse image for icon
    # app.setWindowIcon(get_image_icon("pulse.png"))
    window = pulse_window(cfg)
    window.activateWindow()
    window.show()
    ex = app.exec_()
    del app  # prevents unwanted segfault on closing the window
    sys.exit(ex)
