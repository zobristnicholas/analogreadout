#! /usr/bin/env python

import os
import sys
import numpy as np
from functools import partialmethod
from pymeasure.display.Qt import QtGui
from analogreadout.daq import DAQ
from analogreadout.procedures import Sweep1, Sweep2
from mkidplotter import (SweepGUI, SweepGUIProcedure1, SweepGUIProcedure2, SweepPlotWidget, NoisePlotWidget,
                         TransmissionPlotWidget, TimePlotIndicator, get_image_icon)
         
import pulse_gui                


def open_pulse_gui(self, experiment, bias=None, config=None):
    # Only make pulse gui if it hasn't been opened or was closed
    if self.pulse_window is None or not self.pulse_window.isVisible():
        self.pulse_window = pulse_gui.pulse_window(config=config)
        # make sure pulse window can see sweep window for properly closing daq
        self.pulse_window.sweep_window = self
    # set pulse window inputs to the current experiment values
    sweep_parameters = experiment.procedure.parameter_objects()
    pulse_parameters = self.pulse_window.make_procedure().parameter_objects()
    for key, value in sweep_parameters.items():
        if key in pulse_parameters.keys():
            pulse_parameters[key] = sweep_parameters[key]
    # set the sweep file
    directory = experiment.procedure.directory
    file_name = experiment.data_filename
    file_path = os.path.join(directory, file_name)
    pulse_parameters['sweep_file'].value = file_path
    # set the frequency parameters to the fits instead of the sweep center
    if bias is not None:
        npz_file = np.load(file_path)
        noise_bias = npz_file['noise_bias']
        for b in bias:  # fit might not have been done
            if noise_bias[b[0]]:
                pulse_parameters[b[1]].value = noise_bias[b[0]]
    self.pulse_window.inputs.set_parameters(pulse_parameters)
    # show the window
    self.pulse_window.activateWindow()
    self.pulse_window.show()


def sweep_window(config="UCSB"):
    # setup options
    if config == "UCSB":
        x_list = (('i1', 'i1_bias'), ('f1',), ('f1_psd', 'f1_psd'), ('i2', 'i2_bias'), ('f2',), ('f2_psd', 'f2_psd'))
        y_list = (('q1', 'q1_bias'), ('t1',), ("i1_psd", "q1_psd"), ('q2', 'q2_bias'), ('t2',), ("i2_psd", "q2_psd"))
        x_label = ("I [V]", "frequency [GHz]", "frequency [Hz]", "I [V]", "frequency [GHz]", "frequency [Hz]")
        y_label = ("Q [V]", "|S21| [dBm]", "PSD [V² / Hz]", "Q [V]", "|S21| [dBm]", "PSD [V² / Hz]")
        legend_list = (('loop', 'bias point'), None, ('I', 'Q'),
                       ('loop', 'bias point'), None, ('I', 'Q'))
        widgets_list = (SweepPlotWidget, TransmissionPlotWidget, NoisePlotWidget,
                        SweepPlotWidget, TransmissionPlotWidget, NoisePlotWidget)
        names_list = ('Channel 1: IQ', 'Channel 1: |S21|', 'Channel 1: Noise',
                      'Channel 2: IQ', 'Channel 2: |S21|', 'Channel 2: Noise')
        procedure_class = Sweep2
        base_procedure_class = SweepGUIProcedure2
        bias = [(0, "frequency1"), (3, "frequency2")]
    elif config == "JPL":
        x_list = (('i', 'i_bias'), ('f',), ('f_psd', 'f_psd'))
        y_list = (('q', 'q_bias'), ('t',), ("i_psd", "q_psd"))
        x_label = ("I [V]", "frequency [GHz]", "frequency [Hz]")
        y_label = ("Q [V]", "|S21| [dBm]", "PSD [V² / Hz]")
        legend_list = (('loop', 'bias point'), None, ('I', 'Q'))
        widgets_list = (SweepPlotWidget, TransmissionPlotWidget, NoisePlotWidget)
        names_list = ('IQ', '|S21|', 'Noise')
        procedure_class = Sweep1
        base_procedure_class = SweepGUIProcedure1
        bias = [(0, "frequency")]
    else:
        raise ValueError("'{}' is not a valid configuration".format(config))
    # setup indicators
    indicators = TimePlotIndicator(pulse_gui.time_stamps, pulse_gui.temperatures, title='Device Temperature [mK]')
    pulse_gui.temperature_updater = pulse_gui.Updater()
    # patch the function to open the pulse gui
    SweepGUI.open_pulse_gui = partialmethod(open_pulse_gui, bias=bias, config=config)
    # make the window
    w = SweepGUI(procedure_class, base_procedure_class=base_procedure_class, x_axes=x_list,
                 y_axes=y_list, x_labels=x_label, y_labels=y_label,
                 legend_text=legend_list, plot_widget_classes=widgets_list,
                 plot_names=names_list, log_level="INFO", persistent_indicators=indicators)
    # connect the daq to the process after making the window so that the log widget gets
    # the instrument creation log messages
    pulse_gui.daq = DAQ(config)
    procedure_class.connect_daq(pulse_gui.daq)
    return w


if __name__ == '__main__':
    pulse_gui.setup_logging()
    if len(sys.argv) > 1:
        cfg = sys.argv.pop(1)
    else:
        cfg = "UCSB"
    app = QtGui.QApplication(sys.argv)
    app.setWindowIcon(get_image_icon("loop.png"))
    window = sweep_window(cfg)
    window.activateWindow()
    window.show()
    ex = app.exec_()
    del app  # prevents unwanted segfault on closing the window
    sys.exit(ex)
