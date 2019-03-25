import os
import sys
import numpy as  np
from pymeasure.display.Qt import QtGui
from analogreadout.daq import DAQ
from analogreadout.procedures import Sweep2
from mkidplotter import (SweepGUI, SweepGUIProcedure2, SweepPlotWidget, NoisePlotWidget, TransmissionPlotWidget,
                         TimePlotIndicator, get_image_icon)
         
import pulse_gui                
# from pulse_gui import pulse_window, temperature, setup_logging, daq, indicators
                         

def open_pulse_gui(self, experiment):
    # Only make pulse gui if it hasn't been opened or was closed
    if self.pulse_window is None or not self.pulse_window.isVisible():
        self.pulse_window = pulse_gui.pulse_window()
        # make sure pulse window can see sweep window for properly closing daq
        self.pulse_window.sweep_window = self
    # set pulse window inputs to the current experiment values
    sweep_parameters = experiment.procedure.parameter_objects()
    pulse_parameters = self.pulse_window.make_procedure().parameter_objects()
    for key, value in sweep_parameters.items():
        if key in pulse_parameters.keys():
            pulse_parameters[key] = sweep_parameters[key]
    # set the frequency parameters to the fits instead of the sweep center
    directory = experiment.procedure.directory
    file_name = experiment.data_filename
    npz_file = np.load(os.path.join(directory, file_name))
    noise_bias = npz_file['noise_bias']
    if noise_bias[0]:  # fit might not have been done
        pulse_parameters['frequency1'].value = noise_bias[0]
    if noise_bias[3]:
        pulse_parameters['frequency2'].value = noise_bias[3]
    self.pulse_window.inputs.set_parameters(pulse_parameters)
    # show the window
    self.pulse_window.activateWindow()
    self.pulse_window.show()
        

def sweep_window():
    # setup options
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
    indicators = TimePlotIndicator(pulse_gui.time_stamps, pulse_gui.temperatures, title='Device Temperature [mK]')
    if pulse_gui.temperature_updator is None:
        pulse_gui.temperature_updator = pulse_gui.TemperatureUpdator()
    # patch the function to open the pulse gui
    SweepGUI.open_pulse_gui = open_pulse_gui    
    # make the window
    w = SweepGUI(Sweep2, base_procedure_class=SweepGUIProcedure2, x_axes=x_list,
                 y_axes=y_list, x_labels=x_label, y_labels=y_label,
                 legend_text=legend_list, plot_widget_classes=widgets_list,
                 plot_names=names_list, log_level="INFO", persistent_indicators=indicators)
    # connect the daq to the process after making the window so that the log widget gets
    # the instrument creation log messages
    pulse_gui.daq = DAQ("UCSB")
    Sweep2.connect_daq(pulse_gui.daq)
    return w


if __name__ == '__main__':
    # TODO: implement JPL/UCSB configuration switching
    pulse_gui.setup_logging()
    app = QtGui.QApplication(sys.argv)
    app.setWindowIcon(get_image_icon("loop.png"))
    window = sweep_window()
    window.activateWindow()
    window.show()
    ex = app.exec_()
    del app  # prevents unwanted segfault on closing the window
    sys.exit(ex)