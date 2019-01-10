import os
import logging
import tempfile
import warnings
import numpy as np
from mkidplotter import (SweepBaseProcedure, MKIDProcedure)
from mkidplotter.gui.parameters import DirectoryParameter
from pymeasure.experiment import (IntegerParameter, FloatParameter, BooleanParameter,
                                  Results)

log = logging.getLogger(__name__)
log.addHandler(logging.NullHandler())


class Sweep(SweepBaseProcedure):
    # outputs
    freqs = None
    z = None
    z_offset = None
    calibration = None

    def execute(self):
        # calibrate the data (if possible)
        self.calibrate()
        # initialize the system in the right mode
        with warnings.catch_warnings():
            # ignoring warnings for setting infinite attenuation
            warnings.simplefilter("ignore", UserWarning)
            self.daq.initialize(self.freqs[:, 0], dac_atten=np.inf, adc_atten=np.inf, 
                                sample_rate=self.sample_rate, n_samples=self.n_samples)                 
        # loop through the frequencies and take data
        for index, _ in enumerate(self.freqs[0, :]):
            self.daq.dac.set_frequency(self.freqs[:, index])
            self.z_offset[:, index] = self.daq.adc.take_iq_point()
            self.send('progress', index / self.n_points * 100 / 2)
            log.debug("taking zero index: %d", index)
            if self.stop():
                log.warning("Caught the stop flag in the procedure")
                return

        # initialize the system in the right mode
        adc_atten = max(0, self.total_atten - self.attenuation)
        self.daq.initialize(self.freqs[:, 0], dac_atten=self.attenuation,
                            adc_atten=adc_atten, sample_rate=self.sample_rate,
                            n_samples=self.n_samples)
        # loop through the frequencies and take data
        for index, _ in enumerate(self.freqs[0, :]):
            self.daq.dac.set_frequency(self.freqs[:, index])
            self.z[:, index] = self.daq.adc.take_iq_point()
            data = self.get_sweep_data(index)
            self.send('results', data)
            self.send('progress', 50 + index / self.n_points * 100 / 2)
            log.debug("taking data index: %d", index)
            if self.stop():
                log.warning("Caught the stop flag in the procedure")
                return

        if self.take_noise:
            file_name = self.file_name().split("_")
            time = "_".join(file_name[-2:]).split(".")[0]
            numbers = []
            for number in file_name[:-2]:
                try:
                    numbers.append(int(number))
                except ValueError:
                    pass
            file_name_kwargs = {"prefix": "noise", "numbers": numbers, "time": time}
            self.daq.run("noise", file_name_kwargs, directory=self.directory,
                         attenuation=self.attenuation)
                 
    def shutdown(self):
        self.save()  # save data even if the procedure was aborted
        log.info("Finished sweep procedure")
        
    def make_procedure_from_file(self, npz_file):
        # load in the data
        metadata = npz_file['metadata'].item()
        parameter_dict = metadata['parameters']
        # make a procedure object with the right parameters
        procedure = self.__class__()
        for name, value in parameter_dict.items():
            setattr(procedure, name, value)
        procedure.refresh_parameters()  # Enforce update of meta data
        return procedure
          
    def make_results(self, records, procedure):
        # make a temporary file for the gui data
        file_path = os.path.abspath(tempfile.mktemp(suffix=".txt"))
        # make the mkidplotter Results class
        results = Results(procedure, file_path)
        log.info("Loading dataset into the temporary file %s", file_path)
        # write the record array to the file
        with open(file_path, mode='a') as temporary_file:
            for index in range(records.shape[0]):
                temporary_file.write(results.format(records[index]))
                temporary_file.write(os.linesep)
        return results
                
    def save(self):
        file_path = os.path.join(self.directory, self.file_name())
        log.info("Saving data to %s", file_path)
        np.savez(file_path, freqs=self.freqs, z=self.z, z_offset=self.z_offset,
                 calibration=self.calibration, metadata=self.metadata)
                 
    def load(self, file_path):
        """
        Load the procedure output into a pymeasure Results class instance for the GUI.
        """
        # load in the data
        npz_file = np.load(file_path)
        # create empty numpy structured array
        procedure = self.make_procedure_from_file(npz_file)
        # make array with data
        records = self.make_record_array(npz_file)
        # TODO: add noise to record array
        # make a temporary file for the gui data
        results = self.make_results(records, procedure)
        return results
        
    def startup(self):
        pass
    
    def calibrate(self):
        pass
    
    def get_sweep_data(self, index):
        raise NotImplementedError
        
    def make_record_array(self, npz_file):
        raise NotImplementedError


class Sweep1(Sweep):
    # parameters
    frequency = FloatParameter("Center Frequency", units="GHz", default=4.0)
    span = FloatParameter("Span", units="MHz", default=2)
    sample_rate = FloatParameter("Sample Rate", units="Hz", default=2e6)
    n_samples = IntegerParameter("Samples to Average", default=1000)
    take_noise = BooleanParameter("Take Noise Data", default=True)
    n_points = IntegerParameter("Number of Points", default=500)
    total_atten = FloatParameter("Total Attenuation", units="dB", default=0)
    # gui data columns
    DATA_COLUMNS = ['f', 'i', 'q', 'i_bias', 'q_bias', 'i_psd', 'q_psd', 'f_psd']
    
    def startup(self):
        log.info("Starting sweep procedure")
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

    def make_record_array(self, npz_file):
        # create empty numpy structured array
        size =npz_file["freqs"][0, :].size
        dt = list(zip(self.DATA_COLUMNS, [float] * len(self.DATA_COLUMNS)))
        records = np.empty((size,), dtype=dt)
        records.fill(np.nan)
        # fill array
        records["f"] = npz_file["freqs"][0, :] 
        records["i"] = npz_file["z"][0, :].real - npz_file["z_offset"][0, :].real
        records["q"] = npz_file["z"][0, :].imag - npz_file["z_offset"][0, :].imag
        return records
        

class Sweep2(Sweep):
    # parameters
    frequency1 = FloatParameter("Channel 1 Center Frequency", units="GHz", default=4.0)
    span1 = FloatParameter("Channel 1 Span", units="MHz", default=2)
    frequency2 = FloatParameter("Channel 2 Center Frequency", units="GHz", default=4.0)
    span2 = FloatParameter("Channel 2 Span", units="MHz", default=2)
    sample_rate = FloatParameter("Sample Rate", units="Hz", default=8e5)
    n_samples = IntegerParameter("Samples to Average", default=20000)
    take_noise = BooleanParameter("Take Noise Data", default=True)
    n_points = IntegerParameter("Number of Points", default=500)
    total_atten = FloatParameter("Total Attenuation", units="dB", default=0)
    # gui data columns
    DATA_COLUMNS = ['f1', 'i1', 'q1', 't1', 'i1_bias', 'q1_bias', 'i1_psd', 'q1_psd',
                    'f2', 'i2', 'q2', 't2', 'i2_bias', 'q2_bias', 'i2_psd', 'q2_psd',
                    'f_psd']

    def startup(self):
        log.info("Starting sweep procedure")
        # create output data structures so that data is still saved after abort
        self.freqs = np.vstack(
            (np.linspace(self.frequency1 - self.span1 / 2,
                         self.frequency1 + self.span1 / 2, self.n_points),
             np.linspace(self.frequency2 - self.span2 / 2,
                         self.frequency2 + self.span2 / 2, self.n_points)))
        self.freqs = np.round(self.freqs, 9)  # round to nearest Hz 
        self.z = np.zeros(self.freqs.shape, dtype=np.complex)
        self.z_offset = np.zeros(self.freqs.shape, dtype=np.complex)
        self.calibration = np.zeros((2, 3, self.daq.adc.samples_per_channel),
                                    dtype=np.complex)
        # save parameter metadata
        self.update_metadata()
        # TODO: set temperature and aux field here and wait for stabilization

    def get_sweep_data(self, index):
        dB0 = 10 * np.log10(1e-3 / 50)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            t1 = 20 * np.log10(np.abs(self.z[0, index] - self.z_offset[0, index])) - dB0
            t2 = 20 * np.log10(np.abs(self.z[1, index] - self.z_offset[1, index])) - dB0
        t1 = np.nan if np.isinf(t1) else t1
        t2 = np.nan if np.isinf(t2) else t2    
        data = {"f1": self.freqs[0, index],
                "i1": self.z[0, index].real - self.z_offset[0, index].real,
                "q1": self.z[0, index].imag - self.z_offset[0, index].imag,
                "t1": t1,
                "f2": self.freqs[1, index],
                "i2": self.z[1, index].real - self.z_offset[1, index].real,
                "q2": self.z[1, index].imag - self.z_offset[1, index].imag,
                "t2": t2}
        return data
    
    def make_record_array(self, npz_file):
        # create empty numpy structured array
        size =npz_file["freqs"][0, :].size
        dt = list(zip(self.DATA_COLUMNS, [float] * len(self.DATA_COLUMNS)))
        records = np.empty((size,), dtype=dt)
        records.fill(np.nan)
        # fill array
        dB0 = 10 * np.log10(1e-3 / 50)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            t1 = 20 * np.log10(np.abs(
                npz_file["z"][0, :] - npz_file["z_offset"][0, :])) - dB0
            t2 = 20 * np.log10(np.abs(
                npz_file["z"][1, :] - npz_file["z_offset"][1, :])) - dB0
        t1[np.isinf(t1)] = np.nan
        t2[np.isinf(t2)] = np.nan
                    
        records["f1"] = npz_file["freqs"][0, :] 
        records["i1"] = npz_file["z"][0, :].real - npz_file["z_offset"][0, :].real
        records["q1"] = npz_file["z"][0, :].imag - npz_file["z_offset"][0, :].imag
        records["t1"] = t1
        records["f2"] = npz_file["freqs"][1, :]
        records["i2"] = npz_file["z"][1, :].real - npz_file["z_offset"][1, :].real
        records["q2"] = npz_file["z"][1, :].imag - npz_file["z_offset"][1, :].imag
        records["t2"] = t2
        return records
        
    def calibrate(self):
        # initialize in noise data mode
        adc_atten = max(0, self.total_atten - self.attenuation)
        self.daq.initialize("noise_data", self.freqs[:, 0],
                            dac_atten=self.attenuation, adc_atten=adc_atten)
        # channel 1 lowest frequency
        self.daq.dac.set_frequency([self.freqs[0, 0], self.freqs[0, 0] + 1e-5])
        self.calibration[0, 0, :] = self.daq.adc.take_noise_data(1)[0]
        # channel 2 lowest frequency
        self.daq.dac.set_frequency([self.freqs[1, 0] + 1e-5, self.freqs[1, 0]])
        self.calibration[1, 0, :] = self.daq.adc.take_noise_data(1)[1]
        # channel 1 middle frequency
        self.daq.dac.set_frequency([self.frequency1, self.frequency1 + 1e-5])
        self.calibration[0, 1, :] = self.daq.adc.take_noise_data(1)[0]
        # channel 2 middle frequency
        self.daq.dac.set_frequency([self.frequency2 + 1e-5, self.frequency2])
        self.calibration[1, 1, :] = self.daq.adc.take_noise_data(1)[1]
        # channel 1 highest frequency
        self.daq.dac.set_frequency([self.freqs[0, -1], self.freqs[0, -1] + 1e-5])
        self.calibration[0, 2, :] = self.daq.adc.take_noise_data(1)[0]
        # channel 2 highest frequency
        self.daq.dac.set_frequency([self.freqs[1, -1] + 1e-5, self.freqs[1, -1]])
        self.calibration[1, 2, :] = self.daq.adc.take_noise_data(1)[1]


class Noise(MKIDProcedure):
    # outputs
    freqs = None
    noise = None

    def execute(self):
        adc_atten = max(0, self.total_atten - self.attenuation)
        n_samples = int(self.time * self.sample_rate)
        for index, _ in enumerate(self.freqs[0, :]):
            # initialize the system in the right mode
            self.daq.initialize(self.freqs[:, index], dac_atten=self.attenuation,
                                adc_atten=adc_atten, sample_rate=self.sample_rate,
                                n_samples=n_samples)
            # take the data
            data = self.daq.adc.take_noise_data(self.n_integrations)
            self.noise[:, index, :, :] = data   
            if self.stop():
                log.warning("Caught the stop flag in the procedure")
                return
            
    def shutdown(self):
        self.save()  # save data even if the procedure was aborted
        log.info("Finished noise procedure")
        
    def save(self):
        file_path = os.path.join(self.directory, self.file_name())
        log.info("Saving data to %s", file_path)
        np.savez(file_path, freqs=self.freqs, noise=self.noise, metadata=self.metadata)

  
class Noise2(Noise):
    frequency1 = FloatParameter("Channel 1 Bias Frequency", units="GHz", default=4.0)
    frequency2 = FloatParameter("Channel 2 Bias Frequency", units="GHz", default=4.0)
    sample_rate = FloatParameter("Sampling Rate", units="Hz", default=8e5)
    # TODO: make custom parameter for this
    off_res = BooleanParameter("Take Off Resonance Data", default=True)
    time = FloatParameter("Integration Time", default=1)
    n_integrations = IntegerParameter("Number of Integrations", default=1)
    attenuation = FloatParameter("DAC Attenuation", units="dB")
    total_atten = IntegerParameter("Total Attenuation", units="dB", default=0) 
    directory = DirectoryParameter("Data Directory")
    
    def startup(self):
        log.info("Starting noise procedure")
        # create output data structures so that data is still saved after abort
        n_noise = 1 + self.off_res  # TODO: update when off_res param is updated
        n_points = int(self.time * self.sample_rate)
        self.freqs = np.zeros((2, n_noise))
        self.freqs[:, 0] = np.array([self.frequency1, self.frequency2])
        # TODO: un-hardcode
        self.freqs[:, 1] = np.array([self.frequency1 + 0.002, self.frequency2 + 0.002])
        self.noise = np.zeros((2, n_noise, self.n_integrations, n_points),
                              dtype=np.complex)
        self.update_metadata()