import os
import logging
import tempfile
import warnings
import numpy as np
from mkidplotter import SweepBaseProcedure
from pymeasure.experiment import (IntegerParameter, FloatParameter, BooleanParameter,
                                  Results)

log = logging.getLogger(__name__)
log.addHandler(logging.NullHandler())


class Sweep(SweepBaseProcedure):
    # daq will be connected to the class with connect_daq() class method
    daq = None
    # outputs
    freqs = None
    z = None
    z_offset = None
    metadata = {"parameters": {}}

    def execute(self):
        # TODO: take calibration data
        # initialize the system in the right mode
        with warnings.catch_warnings():
            # ignoring warnings for setting infinite attenuation
            warnings.simplefilter("ignore", UserWarning)
            self.daq.initialize("sweep_data", self.freqs[:, 0], 
                                dac_atten=np.inf, adc_atten=np.inf)                 
        # loop through the frequencies and take data
        for index, _ in enumerate(self.freqs[0, :]):
            self.daq.dac.set_frequency(self.freqs[:, index])
            self.z_offset[:, index] = self.daq.adc.take_iq_point()
            self.emit('progress', index / self.n_points * 100 / 2)
            log.debug("taking zero index: %d", index)

        # initialize the system in the right mode
        adc_atten = max(0, self.total_atten - self.attenuation)
        self.daq.initialize("sweep_data", self.freqs[:, 0],
                            dac_atten=self.attenuation, adc_atten=adc_atten)
        # loop through the frequencies and take data
        for index, _ in enumerate(self.freqs[0, :]):
            self.daq.dac.set_frequency(self.freqs[:, index])
            self.z[:, index] = self.daq.adc.take_iq_point()
            data = self.get_sweep_data(index)
            self.emit('results', data)
            self.emit('progress', 50 + index / self.n_points * 100 / 2)
            log.debug("taking data index: %d", index)

        if self.take_noise:
            # TODO: make this work
            pass
                 
    def shutdown(self):
        self.save()  # save data even if the procedure was aborted
        log.info("Finished procedure")
        
    @classmethod
    def connect_daq(cls, daq):
        """Connects all current and future instances of the procedure class to the daq"""
        cls.daq = daq
        
    def update_metadata(self):
        # save current parameters
        for name in dir(self):
            if name in self._parameters.keys():
                value = getattr(self, name)
                log.info("Parameter {}: {}".format(name, value))
                self.metadata['parameters'][name] = value
        # save some data from the current state of the daq sensors
        self.metadata.update(self.daq.system_state())
        # save the file name
        self.metadata["file_name"] = self.file_name()

    @staticmethod
    def compute_psd(data):
        pass
        
    def make_procedure_from_file(self, npz_file):
        # load in the data
        parameter_dict = npz_file['parameters'].item()
        # make a procedure object with the right parameters
        procedure = self.__class__()
        for name, value in parameter_dict.items():
            setattr(procedure, name, value)
        procedure.refresh_parameters()  # Enforce update of meta data
        return procedure
        
    def make_structured_array(self, npz_file):
        # find the max size of the data (assumes largest axis of all numpy arrays)
        sizes = []
        for _, value in npz_file.items():
            if hasattr(value, "shape") and value.shape:
                sizes.append(np.max(value.shape))
            else:
                sizes.append(np.max(np.array([value]).shape))
        size = max(sizes)
        # create empty numpy structured array
        dt = [tuple(zip(self.DATA_COLUMNS, [float] * len(self.DATA_COLUMNS)))]
        records = np.empty((size,), dtype=dt)
        records.fill(np.nan)
        return records
        
    def make_results(self, records, procedure):
        # make a temporary file for the gui data
        file_path = tempfile.mktemp()
        # make the mkidplotter Results class
        results = Results(procedure, file_path)
        log.info("Loading dataset into the temporary file %s", file_path)
        # write the record array to the file
        with open(file_path, mode='a') as temporary_file:
            for index in range(records.shape[0]):
                temporary_file.write(results.format(records[index]))
                temporary_file.write(os.linesep)
                
    def save(self):
        file_path = os.path.join(self.directory, self.file_name())
        log.info("Saving data to %s", file_path)
        np.savez(file_path, freqs=self.freqs, z=self.z, z_offset=self.z_offset,
                 metadata=self.metadata)
                 
    def load(self, file_path):
        """
        Load the procedure output into a pymeasure Results class instance for the GUI.
        """
        # load in the data
        npz_file = np.load(file_path)
        # create empty numpy structured array
        records = self.make_structured_array(npz_file)
        procedure = self.make_procedure_from_file(npz_file)
        # fill array
        self.fill_record_array(records, npz_file)
        # TODO: add noise to record array
        # make a temporary file for the gui data
        results = self.make_results(records, procedure)
        return results
        
    def startup(self):
        pass
    
    def get_sweep_data(self, index):
        raise NotImplementedError
        
    @staticmethod
    def fill_record_array(records, npz_file):
        raise NotImplementedError


class Sweep1(Sweep):
    # parameters
    frequency = FloatParameter("Center Frequency", units="GHz", default=4.0)
    span = FloatParameter("Span", units="MHz", default=2)
    take_noise = BooleanParameter("Take Noise Data", default=True)
    n_points = IntegerParameter("Number of Points", default=500)
    total_atten = IntegerParameter("Total Attenuation", units="dB", default=0)
    # gui data columns
    DATA_COLUMNS = ['f', 'i', 'q', 'i_bias', 'q_bias', 'i_psd', 'q_psd', 'f_psd']
    
    def startup(self):
        log.info("Starting procedure:")
        # create output data structures so that data is still saved after abort
        self.freqs = np.atleast_2d(np.linspace(self.frequency - self.span / 2,
                                               self.frequency + self.span / 2,
                                               self.n_points))
        self.freqs = np.round(self.freqs, 9)  # round to nearest Hz        
        self.z = np.zeros(self.freqs.shape, dtype=np.complex)
        self.z_offset = np.zeros(self.freqs.shape, dtype=np.complex)
        # save parameter metadata
        self.update_metadata()
        # TODO: set temperature and aux field here and wait for stabilization
    
    def get_sweep_data(self, index):
        data = {"f1": self.freqs[0, index],
                "i": self.z[0, index].real - self.z_offset[0, index].real,
                "q": self.z[0, index].imag - self.z_offset[0, index].imag}
        return data

    @staticmethod
    def fill_record_array(records, npz_file):
        records["f"] = npz_file["freqs"][0, :] 
        records["i"] = npz_file["z"][0, :].real - npz_file["z_offset"][0, :].real
        records["q"] = npz_file["z"][0, :].imag - npz_file["z_offset"][0, :].imag
        

class Sweep2(Sweep):
    # parameters
    frequency1 = FloatParameter("Channel 1 Center Frequency", units="GHz", default=4.0)
    span1 = FloatParameter("Channel 1 Span", units="MHz", default=2)
    frequency2 = FloatParameter("Channel 2 Center Frequency", units="GHz", default=4.0)
    span2 = FloatParameter("Channel 2 Span", units="MHz", default=2)
    take_noise = BooleanParameter("Take Noise Data", default=True)
    n_points = IntegerParameter("Number of Points", default=500)
    total_atten = IntegerParameter("Total Attenuation", units="dB", default=0)
    # gui data columns
    DATA_COLUMNS = ['f1', 'i1', 'q1', 'i1_bias', 'q1_bias', 'i1_psd', 'q1_psd',
                    'f2', 'i2', 'q2', 'i2_bias', 'q2_bias', 'i2_psd', 'q2_psd', 'f_psd']

    def startup(self):
        log.info("Starting procedure:")
        # create output data structures so that data is still saved after abort
        self.freqs = np.vstack(
            (np.linspace(self.frequency1 - self.span1 / 2,
                         self.frequency1 + self.span1 / 2, self.n_points),
             np.linspace(self.frequency2 - self.span2 / 2,
                         self.frequency2 + self.span2 / 2, self.n_points)))
        self.freqs = np.round(self.freqs, 9)  # round to nearest Hz 
        self.z = np.zeros(self.freqs.shape, dtype=np.complex)
        self.z_offset = np.zeros(self.freqs.shape, dtype=np.complex)
        # save parameter metadata
        self.update_metadata()
        # TODO: set temperature and aux field here and wait for stabilization

    def get_sweep_data(self, index):
        data = {"f1": self.freqs[0, index],
                "i1": self.z[0, index].real - self.z_offset[0, index].real,
                "q1": self.z[0, index].imag - self.z_offset[0, index].imag,
                "f2": self.freqs[1, index],
                "i2": self.z[1, index].real - self.z_offset[1, index].real,
                "q2": self.z[1, index].imag - self.z_offset[1, index].imag}
        return data
    
    @staticmethod
    def fill_record_array(records, npz_file):
        records["f1"] = npz_file["freqs"][0, :] 
        records["i1"] = npz_file["z"][0, :].real - npz_file["z_offset"][0, :].real
        records["q1"] = npz_file["z"][0, :].imag - npz_file["z_offset"][0, :].imag
        records["f2"] = npz_file["freqs"][1, :]
        records["i2"] = npz_file["z"][1, :].real - npz_file["z_offset"][1, :].real
        records["q2"] = npz_file["z"][1, :].imag - npz_file["z_offset"][1, :].imag
