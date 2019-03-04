import os
import re
import sys
import logging
from logging.handlers import TimedRotatingFileHandler
from datetime import datetime
from pymeasure.display.Qt import QtGui
from analogreadout.daq import DAQ
from analogreadout.procedures import Sweep2
from mkidplotter import (SweepGUI, SweepGUIProcedure2, SweepPlotWidget, NoisePlotWidget,
                         TransmissionPlotWidget, get_image_icon)

def setup_logging():
    log = logging.getLogger()
    directory = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'logs'))
    if not os.path.isdir(directory):
        os.mkdir(directory)
    file_name = datetime.now().strftime("sweep_%y%m%d.log")
    file_path = os.path.join(directory, file_name)
    handler = TimedRotatingFileHandler(file_path, when="midnight")
    handler.suffix = "%Y%m%d"
    handler.extMatch = re.compile(r"^\d{8}$")
    handler.setLevel("INFO")
    log_format = logging.Formatter(fmt='%(asctime)s : %(message)s (%(levelname)s)',
                                   datefmt='%I:%M:%S %p')
    handler.setFormatter(log_format)
    log.addHandler(handler)
    return log


def sweep_window():
    x_list = (('i1', 'i1_bias'), ('f1',), ('f_psd', 'f_psd'),
              ('i2', 'i2_bias'), ('f2',), ('f_psd', 'f_psd'))
    y_list = (('q1', 'q1_bias'), ('t1',), ("i1_psd", "q1_psd"),
              ('q2', 'q2_bias'), ('t2',), ("i2_psd", "q2_psd"))
    x_label = ("I [V]", "frequency [GHz]", "frequency [Hz]",
               "I [V]", "frequency [GHz]", "frequency [Hz]")
    y_label = ("Q [V]", "|S21| [dBm]", "PSD [V² / Hz]",
               "Q [V]", "|S21| [dBm]", "PSD [V² / Hz]")
    legend_list = (('loop', 'bias point'), None, ('I', 'Q'),
                   ('loop', 'bias point'), None, ('I', 'Q'))
    widgets_list = (SweepPlotWidget, TransmissionPlotWidget, NoisePlotWidget,
                    SweepPlotWidget, TransmissionPlotWidget, NoisePlotWidget)
    names_list = ('Channel 1: IQ', 'Channel 1: |S21|', 'Channel 1: Noise',
                  'Channel 2: IQ', 'Channel 2: |S21|', 'Channel 2: Noise')
    
    w = SweepGUI(Sweep2, base_procedure_class=SweepGUIProcedure2, x_axes=x_list,
                 y_axes=y_list, x_labels=x_label, y_labels=y_label,
                 legend_text=legend_list, plot_widget_classes=widgets_list,
                 plot_names=names_list, log_level="INFO")
    # connect the daq to the process after making the window so that the log widget gets
    # the instrument creation log messages
    Sweep2.connect_daq(DAQ("UCSB"))
    return w


if __name__ == '__main__':
    # TODO: implement JPL/UCSB configuration switching
    setup_logging()
    app = QtGui.QApplication(sys.argv)
    app.setWindowIcon(get_image_icon("loop.png"))
    window = sweep_window()
    window.activateWindow()
    window.show()
    ex = app.exec_()
    del app  # prevents unwanted segfault on closing the window
    sys.exit(ex)