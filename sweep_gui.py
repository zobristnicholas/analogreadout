import os
import sys
import yaml
import numpy as np
from functools import partialmethod
from pymeasure.display.Qt import QtGui
from analogreadout.daq import DAQ
from mkidplotter import SweepGUI, get_image_icon, TimePlotIndicator
         
import fit_gui
import pulse_gui


def open_pulse_gui(self, experiment, bias=None, configuration=None):
    # Only make pulse gui if it hasn't been opened or was closed
    if self.pulse_window is None or not self.pulse_window.isVisible():
        self.pulse_window = pulse_gui.pulse_window(configuration=configuration)
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


def open_fit_gui(self, experiment, configuration=None):
    # Only make fit gui if it hasn't been opened or was closed
    if self.fit_window is None or not self.fit_window.isVisible():
        self.fit_window = fit_gui.fit_window(configuration=configuration)
    fit_parameters = self.fit_window.make_procedure().parameter_objects()

    # set the sweep file and directory
    directory = experiment.procedure.directory
    file_name = experiment.data_filename
    file_path = os.path.join(directory, file_name)
    fit_parameters['directory'].value = directory
    fit_parameters['sweep_file'].value = file_path
    self.fit_window.inputs.set_parameters(fit_parameters)

    # show the window
    self.fit_window.activateWindow()
    self.fit_window.show()


def sweep_window(configuration):
    # Start the temperature updater
    pulse_gui.temperature_updater = pulse_gui.Updater()
    indicators = TimePlotIndicator(pulse_gui.time_stamps, pulse_gui.temperatures, title='Device Temperature [mK]')

    # patch the SweepGUI class to open the pulse gui
    bias = config['gui']['bias']
    SweepGUI.open_pulse_gui = partialmethod(open_pulse_gui, bias=bias, configuration=configuration)
    SweepGUI.open_fit_gui = partialmethod(open_fit_gui, configuration=configuration)

    # make the window
    w = SweepGUI(persistent_indicators=indicators, **configuration['gui']['sweep'])

    # connect the daq to the process after making the window so that the log widget gets
    # the instrument creation log messages
    pulse_gui.daq = DAQ(configuration)
    configuration['gui']['sweep']['procedure_class'].connect_daq(pulse_gui.daq)
    return w


if __name__ == '__main__':
    # Set up the logging.
    fit_gui.setup_logging()

    # Open the configuration file.
    file_name = sys.argv.pop(1) if len(sys.argv) > 1 else "ucsb"
    file_path = os.path.join(os.getcwd(), file_name)
    print(file_path)
    if not os.path.isfile(file_path):  # check configurations folder
        file_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                 'analogreadout', 'configurations', file_name.lower() + ".yaml")
    with open(file_path, "r") as f:
        config = yaml.load(f, Loader=yaml.Loader)

    # Create the window.
    app = QtGui.QApplication(sys.argv)
    app.setWindowIcon(get_image_icon("loop.png"))
    window = sweep_window(config)
    window.activateWindow()
    window.show()
    ex = app.exec_()
    del app  # prevents unwanted segfault on closing the window
    sys.exit(ex)
