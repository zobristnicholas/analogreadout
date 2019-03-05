import os
import logging
import tempfile
import warnings
import numpy as np
import scipy.signal as sig
from mkidplotter import (SweepBaseProcedure, MKIDProcedure, NoiseInput, Results,
                         DirectoryParameter)
from pymeasure.experiment import (IntegerParameter, FloatParameter, BooleanParameter,
                                  VectorParameter)

log = logging.getLogger(__name__)
log.addHandler(logging.NullHandler())

STOP_WARNING = "Caught the stop flag in the '{}' procedure"


class Sweep(SweepBaseProcedure):
    # outputs
    freqs = None
    z = None
    z_offset = None
    calibration = None
    noise_bias = None

    sample_rate = FloatParameter("Sample Rate", units="MHz", default=0.8)
    n_samples = IntegerParameter("Samples to Average", default=20000)
    n_points = IntegerParameter("Number of Points", default=500)
    total_atten = FloatParameter("Total Attenuation", units="dB", default=0)
    reverse_sweep = BooleanParameter("Reverse Sweep Direction", default=False)
    wait_temp_min = IntegerParameter("Set Temperature Minimum Wait Time", units="minutes", default=0)
    wait_temp_max = IntegerParameter("Set Temperature Maximum Wait Time", units="minutes", default=0)
    noise = VectorParameter("Noise", length=6, default=[1, 1, 10, 1, -1, 10],
                            ui_class=NoiseInput)

    def execute(self):
        # TODO: set_field when there's an instrument hooked up
        self.daq.thermometer.set_temperature(self.temperature, min_wait=self.wait_temp_min, max_wait=self.wait_temp_max)
        # calibrate the data (if possible)
        self.calibrate()
        # initialize the system in the right mode
        with warnings.catch_warnings():
            # ignoring warnings for setting infinite attenuation
            warnings.simplefilter("ignore", UserWarning)
            self.daq.initialize(self.freqs[:, 0], dac_atten=np.inf, adc_atten=np.inf, 
                                sample_rate=self.sample_rate * 1e6,
                                n_samples=self.n_samples)  
        # loop through the frequencies and take data
        for index, _ in enumerate(self.freqs[0, :]):
            self.daq.dac.set_frequency(self.freqs[:, index])
            if index == 0:
                self.daq.adc.take_iq_point()  # first data point is sometimes garbage
            self.z_offset[:, index] = self.daq.adc.take_iq_point()
            self.emit('progress', index / self.n_points * 100 / 2)
            log.debug("taking zero index: %d", index)
            if self.should_stop():
                log.warning(STOP_WARNING.format(self.__class__.__name__))
                return
        # initialize the system in the right mode
        adc_atten = max(0, self.total_atten - self.attenuation)
        self.daq.initialize(self.freqs[:, 0], dac_atten=self.attenuation,
                            adc_atten=adc_atten, sample_rate=self.sample_rate * 1e6,
                            n_samples=self.n_samples)
        # loop through the frequencies and take data
        for index, _ in enumerate(self.freqs[0, :]):
            self.daq.dac.set_frequency(self.freqs[:, index])
            self.z[:, index] = self.daq.adc.take_iq_point()
            data = self.get_sweep_data(index)
            self.emit('results', data)
            self.emit('progress', 50 + index / self.n_points * 100 / 2)
            log.debug("taking data index: %d", index)
            if self.should_stop():
                log.warning(STOP_WARNING.format(self.__class__.__name__))
                return
        # record system state after data taking
        self.metadata.update(self.daq.system_state())
        # take noise data
        if self.noise[0]:
            # get file name kwargs from file_name
            file_name = self.file_name().split("_")
            time = "_".join(file_name[-2:]).split(".")[0]
            numbers = []
            for number in file_name[:-2]:
                try:
                    numbers.append(int(number))
                except ValueError:
                    pass
            file_name_kwargs = {"prefix": "noise", "numbers": numbers, "time": time}
            # get noise kwargs
            noise_kwargs = self.noise_kwargs()
            # run noise procedure
            self.daq.run("noise", file_name_kwargs, should_stop=self.should_stop,
                         emit=self.emit, **noise_kwargs)

    def shutdown(self):
        self.save()  # save data even if the procedure was aborted
        self.clean_up()  # delete references to data so that memory isn't hogged
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

    @staticmethod
    def make_results(records, procedure):
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
                 calibration=self.calibration, noise_bias=self.noise_bias,
                 metadata=self.metadata)
                 
    def clean_up(self):
        self.freqs = None
        self.z = None
        self.z_offset = None
        self.calibration = None
        self.noise_bias = None
        self.metadata = {"parameters": {}}
                 
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
        # make a temporary file for the gui data
        results = self.make_results(records, procedure)
        return results
        
    @staticmethod
    def get_psd(file_path):
        npz_file = np.load(file_path)
        f_psd = npz_file["f_psd"]
        i_psd = npz_file["psd"].real
        q_psd = npz_file["psd"].imag
        return f_psd, i_psd, q_psd
        
    def startup(self):
        pass
    
    def calibrate(self):
        pass
    
    def get_sweep_data(self, index):
        raise NotImplementedError
        
    def make_record_array(self, npz_file):
        raise NotImplementedError
        
    def noise_kwargs(self):
        raise NotImplementedError


class Sweep1(Sweep):
    # special parameters
    frequency = FloatParameter("Center Frequency", units="GHz", default=4.0)
    span = FloatParameter("Span", units="MHz", default=2)
    # gui data columns
    DATA_COLUMNS = ['f', 'i', 'q', 'i_bias', 'q_bias', 'f_bias', 't_bias', 'i_psd',
                    'q_psd', 'f_psd']
    
    def startup(self):
        log.info("Starting sweep procedure")
        # create output data structures so that data is still saved after abort
        self.freqs = np.atleast_2d(np.linspace(self.frequency - self.span * 1e-3 / 2,
                                               self.frequency + self.span * 1e-3 / 2,
                                               self.n_points))
        if self.reverse_sweep:
            self.freqs = self.freqs[:, ::-1]                                       
        self.freqs = np.round(self.freqs, 9)  # round to nearest Hz        
        self.z = np.zeros(self.freqs.shape, dtype=np.complex64)
        self.z_offset = np.zeros(self.freqs.shape, dtype=np.complex64)
        # save parameter metadata
        self.update_metadata()
    
    def get_sweep_data(self, index):
        data = {"f1": self.freqs[0, index],
                "i": self.z[0, index].real - self.z_offset[0, index].real,
                "q": self.z[0, index].imag - self.z_offset[0, index].imag}
        return data

    def make_record_array(self, npz_file):
        # create empty numpy structured array
        size = npz_file["freqs"][0, :].size
        dt = list(zip(self.DATA_COLUMNS, [float] * len(self.DATA_COLUMNS)))
        records = np.empty((size,), dtype=dt)
        records.fill(np.nan)
        # fill array
        records["f"] = npz_file["freqs"][0, :] 
        records["i"] = npz_file["z"][0, :].real - npz_file["z_offset"][0, :].real
        records["q"] = npz_file["z"][0, :].imag - npz_file["z_offset"][0, :].imag
        return records
        
    def noise_kwargs(self):
        kwargs = {'directory': self.directory,
                  'attenuation': self.attenuation,
                  'sample_rate': self.sample_rate,
                  'total_atten': self.total_atten,
                  'frequency': self.frequency}
        return kwargs
        

class Sweep2(Sweep):
    # special parameters
    frequency1 = FloatParameter("Ch 1 Frequency", units="GHz", default=4.0)
    span1 = FloatParameter("Ch 1 Span", units="MHz", default=2)
    frequency2 = FloatParameter("Ch 2 Frequency", units="GHz", default=4.0)
    span2 = FloatParameter("Ch 2 Span", units="MHz", default=2)
    # gui data columns
    DATA_COLUMNS = ['f1', 'i1', 'q1', 't1', 'f1_bias', 't1_bias', 'i1_bias', 'q1_bias',
                    'i1_psd', 'q1_psd',
                    'f2', 'i2', 'q2', 't2', 'f2_bias', 't2_bias', 'i2_bias', 'q2_bias',
                    'i2_psd', 'q2_psd', 'f_psd']

    def startup(self):
        log.info("Starting sweep procedure")
        # create output data structures so that data is still saved after abort
        self.freqs = np.vstack(
            (np.linspace(self.frequency1 - self.span1 * 1e-3 / 2,
                         self.frequency1 + self.span1 * 1e-3 / 2, self.n_points),
             np.linspace(self.frequency2 - self.span2 * 1e-3 / 2,
                         self.frequency2 + self.span2 * 1e-3 / 2, self.n_points)))
        if self.reverse_sweep:
            self.freqs = self.freqs[:, ::-1]
        self.freqs = np.round(self.freqs, 9)  # round to nearest Hz 
        self.z = np.zeros(self.freqs.shape, dtype=np.complex64)
        self.z_offset = np.zeros(self.freqs.shape, dtype=np.complex64)
        self.calibration = np.zeros((2, 3, self.n_samples),
                                    dtype=np.complex64)
        self.noise_bias = np.zeros(6)
        # save parameter metadata
        self.update_metadata()

    def get_sweep_data(self, index):
        db0 = 10 * np.log10(1e-3 / 50)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            t1 = 20 * np.log10(np.abs(self.z[0, index] - self.z_offset[0, index])) - db0
            t2 = 20 * np.log10(np.abs(self.z[1, index] - self.z_offset[1, index])) - db0
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
        # get noise data
        try:
            noise_file = os.path.basename(npz_file.fid.name).split("_")
            noise_file[0] = "noise"
            noise_file = "_".join(noise_file)
            noise_file = os.path.join(os.path.dirname(npz_file.fid.name), noise_file)
            noise_file = np.load(noise_file)
            psd = noise_file["psd"]
            freqs = noise_file["f_psd"]
            size2 = freqs[0, :].size
        except FileNotFoundError:
            size2 = 1
            freqs = np.zeros((2, size2)) * np.nan
            psd = np.zeros((2, 1, size2)) * np.nan

        # create empty numpy structured array
        size1 = npz_file["freqs"][0, :].size
        size = max(size1, size2)
        dt = list(zip(self.DATA_COLUMNS, [float] * len(self.DATA_COLUMNS)))
        records = np.empty((size,), dtype=dt)
        records.fill(np.nan)
        # fill array
        db0 = 10 * np.log10(1e-3 / 50)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            t1 = 20 * np.log10(np.abs(
                npz_file["z"][0, :] - npz_file["z_offset"][0, :])) - db0
            t2 = 20 * np.log10(np.abs(
                npz_file["z"][1, :] - npz_file["z_offset"][1, :])) - db0
        t1[np.isinf(t1)] = np.nan
        t2[np.isinf(t2)] = np.nan
        records["f1"][:size1] = npz_file["freqs"][0, :] 
        records["i1"][:size1] = npz_file["z"][0, :].real - npz_file["z_offset"][0, :].real
        records["q1"][:size1] = npz_file["z"][0, :].imag - npz_file["z_offset"][0, :].imag
        records["t1"][:size1] = t1
        records["i1_psd"][:size2] = psd[0, 0, :].real
        records["q1_psd"][:size2] = psd[0, 0, :].imag
        records["f2"][:size1] = npz_file["freqs"][1, :]
        records["i2"][:size1] = npz_file["z"][1, :].real - npz_file["z_offset"][1, :].real
        records["q2"][:size1] = npz_file["z"][1, :].imag - npz_file["z_offset"][1, :].imag
        records["t2"][:size1] = t2
        records["i2_psd"][:size2] = psd[1, 0, :].real
        records["q2_psd"][:size2] = psd[1, 0, :].imag
        records["f_psd"][:size2] = freqs[0, :]
        if not (npz_file["noise_bias"][1] == np.zeros(6)).all():
            records["f1_bias"][:1] = npz_file["noise_bias"][0]
            records["t1_bias"][:1] = 20 * np.log10(np.abs(npz_file["noise_bias"][1] +
                                                          1j * npz_file["noise_bias"][2]))
            records["i1_bias"][:1] = npz_file["noise_bias"][1]
            records["q1_bias"][:1] = npz_file["noise_bias"][2]
            records["f2_bias"][:1] = npz_file["noise_bias"][3]
            records["t2_bias"][:1] = 20 * np.log10(np.abs(npz_file["noise_bias"][4] +
                                                          1j * npz_file["noise_bias"][5]))
            records["i2_bias"][:1] = npz_file["noise_bias"][4]
            records["q2_bias"][:1] = npz_file["noise_bias"][5]

        return records
        
    def noise_kwargs(self):
        # compute resonance frequency
        z = self.z - self.z_offset
        filter_win_length = int(np.round(z.shape[1] / 100.0))
        if filter_win_length % 2 == 0:
            filter_win_length += 1
        if filter_win_length < 3:
            filter_win_length = 3
        z_filtered = (sig.savgol_filter(z.real, filter_win_length, 1) +
                      1j * sig.savgol_filter(z.imag, filter_win_length, 1))
        velocity = np.abs(np.diff(z_filtered))
        indices = np.argmax(velocity, axis=-1)
        df = self.freqs[:, 1] - self.freqs[:, 0]
        frequencies = self.freqs[range(z.shape[0]), indices] + df / 2
        
        self.noise_bias = np.array([frequencies[0],
                                    np.mean(z[0, indices[0]: indices[0] + 2].real),
                                    np.mean(z[0, indices[0]: indices[0] + 2].imag),
                                    frequencies[1],
                                    np.mean(z[1, indices[1]: indices[1] + 2].real),
                                    np.mean(z[1, indices[1]: indices[1] + 2].imag)])
        self.emit("results", {'f1_bias': frequencies[0],
                              't1_bias': 20 * np.log10(np.abs(self.noise_bias[1] +
                                                              1j * self.noise_bias[2])),
                              'i1_bias': self.noise_bias[1],
                              'q1_bias': self.noise_bias[2],
                              'f2_bias': frequencies[1],
                              't2_bias': 20 * np.log10(np.abs(self.noise_bias[4] +
                                                              1j * self.noise_bias[5])),
                              'i2_bias': self.noise_bias[4],
                              'q2_bias': self.noise_bias[5]})
        
        kwargs = {'directory': self.directory,
                  'attenuation': self.attenuation,
                  'sample_rate': self.sample_rate,
                  'total_atten': self.total_atten,
                  'frequency1': frequencies[0],
                  'frequency2': frequencies[1],
                  'time': self.noise[1],
                  'n_integrations': self.noise[2],
                  'off_res': bool(self.noise[3]),
                  'offset': self.noise[4],
                  'n_offset': self.noise[5]}
        return kwargs
        
    def calibrate(self):
        # initialize in noise data mode
        adc_atten = max(0, self.total_atten - self.attenuation)
        self.daq.initialize(self.freqs[:, 0], dac_atten=self.attenuation,
                            adc_atten=adc_atten, sample_rate=self.sample_rate * 1e6,
                            n_samples=self.n_samples)
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
    f_psd = None
    psd = None

    def execute(self):
        adc_atten = max(0, self.total_atten - self.attenuation)
        n_samples = int(self.time * self.sample_rate * 1e6)
        for index, _ in enumerate(self.freqs[0, :]):
            # initialize the system in the right mode
            self.daq.initialize(self.freqs[:, index], dac_atten=self.attenuation,
                                adc_atten=adc_atten, sample_rate=self.sample_rate * 1e6,
                                n_samples=n_samples)
            # take the data
            data = self.daq.adc.take_noise_data(self.n_integrations)
            self.noise[:, index, :, :] = data   
            if self.should_stop():
                log.warning(STOP_WARNING.format(self.__class__.__name__))
                return
        # record system state after data taking
        self.metadata.update(self.daq.system_state())
        # send the data to the gui
        self.compute_psd()
        data = self.get_noise_data()
        self.emit('results', data)
            
    def shutdown(self):
        self.save()  # save data even if the procedure was aborted
        self.clean_up()  # delete references to data so that memory isn't hogged
        log.info("Finished noise procedure")
        
    def save(self):
        file_path = os.path.join(self.directory, self.file_name())
        log.info("Saving data to %s", file_path)
        np.savez(file_path, freqs=self.freqs, noise=self.noise, f_psd=self.f_psd,
                 psd=self.psd, metadata=self.metadata)
                 
    def clean_up(self):
        self.freqs = None
        self.noise = None
        self.f_psd = None
        self.psd = None
        self.metadata = {"parameters": {}}
                 
    def compute_psd(self):
        # n_points such that 100 Hz is the minimum possible freq
        n_points = min(self.noise.shape[-1], int(self.sample_rate * 1e6 / 100))
        kwargs = {'nperseg': n_points, 'fs': self.sample_rate * 1e6,
                  'return_onesided': True, 'detrend': 'constant', 'scaling': 'density',
                  'axis': -1, 'window': 'hanning'}
        _, i_psd = sig.welch(self.noise.real, **kwargs)
        _, q_psd = sig.welch(self.noise.imag, **kwargs)
        # average multiple PSDs together
        i_psd = np.mean(i_psd, axis=-2)
        q_psd = np.mean(q_psd, axis=-2)
        # fix zero point
        i_psd[:, :, 0] = i_psd[:, :, 1]
        q_psd[:, :, 0] = q_psd[:, :, 1]
        # save in array
        self.psd = i_psd + 1j * q_psd

    def get_noise_data(self):
        raise NotImplementedError


class Noise2(Noise):
    directory = DirectoryParameter("Data Directory")
    attenuation = FloatParameter("DAC Attenuation", units="dB")
    sample_rate = FloatParameter("Sampling Rate", units="Hz", default=8e5)
    total_atten = IntegerParameter("Total Attenuation", units="dB", default=0) 
    frequency1 = FloatParameter("Channel 1 Bias Frequency", units="GHz", default=4.0)
    frequency2 = FloatParameter("Channel 2 Bias Frequency", units="GHz", default=4.0)
    time = FloatParameter("Integration Time", default=1)
    n_integrations = IntegerParameter("Number of Integrations", units="s", default=1)
    off_res = BooleanParameter("Take Off Resonance Data", default=True)
    offset = FloatParameter("Frequency Offset", units="MHz", default=-1)
    n_offset = FloatParameter("# of Points", default=10)
    
    def startup(self):
        log.info("Starting noise procedure")
        # create output data structures so that data is still saved after abort
        n_noise = 1 + self.off_res * self.n_offset
        n_points = int(self.time * self.sample_rate * 1e6)
        offset = np.linspace(0, self.offset, self.off_res * self.n_offset + 1)
        self.freqs = np.array([self.frequency1 + offset, self.frequency2 + offset])
        self.noise = np.zeros((2, n_noise, self.n_integrations, n_points),
                              dtype=np.complex64)
                              
        n_points = min(self.noise.shape[-1], int(self.sample_rate * 1e6 / 100))
        fft_freq = np.fft.rfftfreq(n_points, d=1 / (self.sample_rate * 1e6))
        n_fft = fft_freq.size
        self.psd = np.zeros((2, n_noise, n_fft), dtype=np.complex64)
        self.f_psd = np.array([fft_freq, fft_freq])
        self.update_metadata()

    def get_noise_data(self):
        data = {"f_psd": self.freqs[0, :],
                "i1_psd": self.psd[0, 0, :].imag,
                "q1_psd": self.psd[0, 0, :].real,
                "i2_psd": self.psd[1, 0, :].real,
                "q2_psd": self.psd[1, 0, :].imag}
        return data
