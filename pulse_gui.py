import os
import sys
import yaml
import numpy as np
from datetime import datetime
from collections import deque
from threading import Thread, Event
from pymeasure.display.Qt import QtGui
from analogreadout.daq import DAQ
from mkidplotter import (PulseGUI, TimePlotIndicator)
import fit_gui

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


def pulse_window(configuration="ucsb2"):
    # Get the configuration
    file_name = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                             'analogreadout', 'configurations',
                             configuration.lower() + ".yaml")
    with open(file_name, "r") as f:
        config = yaml.load(f, Loader=yaml.Loader)

    # Start the temperature updater
    indicators = TimePlotIndicator(time_stamps, temperatures, title='Device Temperature [mK]')
    global temperature_updater
    temperature_updater = Updater()

    # make the window
    w = PulseGUI(persistent_indicators=indicators, **config['gui']['pulse'])

    # create and connect the daq to the process after making the window so that the log widget gets
    # the instrument creation log messages
    global daq
    if daq is not None:
        config['gui']['pulse']['procedure_class'].connect_daq(daq)
    else:
        daq = DAQ(configuration)
        config['gui']['pulse']['procedure_class'].connect_daq(daq)
    return w


if __name__ == '__main__':
    fit_gui.setup_logging()
    if len(sys.argv) > 1:
        cfg = sys.argv.pop(1)
    else:
        cfg = "ucsb2"
    app = QtGui.QApplication(sys.argv)
    # TODO: add pulse image for icon
    # app.setWindowIcon(get_image_icon("pulse.png"))
    window = pulse_window(cfg)
    window.activateWindow()
    window.show()
    ex = app.exec_()
    del app  # prevents unwanted segfault on closing the window
    sys.exit(ex)
