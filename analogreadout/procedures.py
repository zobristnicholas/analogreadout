import os
import yaml
import logging
import tempfile
import warnings
import lmfit as lm
import numpy as np
import scipy.signal as sig
from scipy.interpolate import interp1d
from scipy.stats import median_abs_deviation
from mkidplotter import (SweepBaseProcedure, MKIDProcedure, NoiseInput, Results, DirectoryParameter, BooleanListInput,
                         Indicator, FloatIndicator, FileParameter, FitProcedure, FitInput, RangeInput)
from pymeasure.experiment import (IntegerParameter, FloatParameter, BooleanParameter,
                                  VectorParameter)
from analogreadout.utils import load

import mkidcalculator as mc

log = logging.getLogger(__name__)
log.addHandler(logging.NullHandler())
DB0 = 10 * np.log10(1e-3 / 50)

STOP_WARNING = "Caught the stop flag in the '{}' procedure"


def get_fit_result(result, name):
    if result.errorbars:
        return [float(result.params[name].value), float(result.params[name].stderr)]
    else:
        return [float(result.params[name].value), 0.0]


def make_procedure_from_file(cls, npz_file):
    # load in the data
    metadata = npz_file['metadata'].item()
    parameter_dict = metadata['parameters']
    # make a procedure object with the right parameters
    procedure = cls()
    for name, value in parameter_dict.items():
        setattr(procedure, name, value)
    procedure.refresh_parameters()  # Enforce update of meta data
    return procedure


def make_results(results_dict, procedure):
    # make a results object
    file_path = os.path.abspath(tempfile.mktemp(suffix='.pickle'))
    results = Results(procedure, file_path)
    # update the data in the results
    results.data = results_dict
    return results


class Sweep(SweepBaseProcedure):
    # outputs
    freqs = None
    z = None
    f_offset = None
    z_offset = None
    calibration = None
    noise_bias = None

    sample_rate = FloatParameter("Sample Rate", units="MHz", default=0.8)
    n_samples = IntegerParameter("Samples to Average", default=20000)
    n_points = IntegerParameter("Number of Points", default=500)
    total_atten = FloatParameter("Total Attenuation", units="dB", default=0)
    reverse_sweep = BooleanParameter("Reverse Sweep Direction", default=False)
    wait_temp = FloatParameter("Set Temperature Wait Time", units="minutes", default=0)
    noise = VectorParameter("Noise", length=6, default=[1, 1, 10, 1, -2, 1], ui_class=NoiseInput)
    status_bar = Indicator("Status")

    def execute(self):
        if self.should_stop():
            log.warning(STOP_WARNING.format(self.__class__.__name__))
            return
        # TODO: set_field when there's an instrument hooked up
        self.status_bar.value = "Setting temperature"
        self.daq.thermometer.set_temperature(self.temperature, wait=self.wait_temp, stop=self.should_stop)
        if self.should_stop():
            log.warning(STOP_WARNING.format(self.__class__.__name__))
            return

        self.status_bar.value = "Calibrating IQ mixer offset"
        # initialize the system in the right mode
        # TODO: properly handle nan frequency input by shutting off the corresponding synthesizer
        adc_atten = max(0, self.total_atten - self.attenuation)
        with warnings.catch_warnings():
            # ignoring warnings for setting infinite attenuation
            warnings.simplefilter("ignore", UserWarning)
            self.daq.initialize(self.freqs[:, 0], dac_atten=np.inf, adc_atten=adc_atten,
                                sample_rate=self.sample_rate * 1e6, n_samples=25 * self.n_samples)
        # loop through the frequencies and take data
        for index, _ in enumerate(self.f_offset[0, :]):
            self.daq.dac.set_frequency(self.f_offset[:, index])
            self.z_offset[:, index] = self.daq.adc.take_iq_point()
            self.emit('progress', index / (self.f_offset.shape[1] + self.freqs.shape[1]) * 100)
            log.debug("taking zero index: %d", index)
            if self.should_stop():
                log.warning(STOP_WARNING.format(self.__class__.__name__))
                return
        # calibrate the data (if possible)
        self.calibrate()
        # initialize the system in the right mode
        self.status_bar.value = "Sweeping"
        self.daq.dac_atten.set_attenuation(self.attenuation)
        self.daq.adc.initialize(sample_rate=self.sample_rate * 1e6, n_samples=self.n_samples)
        # loop through the frequencies and take data
        for index, _ in enumerate(self.freqs[0, :]):
            self.daq.dac.set_frequency(self.freqs[:, index])
            self.z[:, index] = self.daq.adc.take_iq_point()
            data = self.get_sweep_data(index)
            self.emit('results', data)
            self.emit('progress', (self.f_offset.shape[1] + index) /
                                  (self.f_offset.shape[1] + self.freqs.shape[1]) * 100)
            log.debug("taking data index: %d", index)
            if self.should_stop():
                log.warning(STOP_WARNING.format(self.__class__.__name__))
                return
        # record system state after data taking
        self.metadata.update(self.daq.system_state())
        # compute the noise bias (fit the loop) even if we aren't taking noise data
        self.compute_noise_bias()
        # take noise data
        if self.noise[0]:
            # get file name kwargs from file_name
            file_name_kwargs = self.file_name_parts()
            file_name_kwargs["prefix"] = "noise"
            # get noise kwargs
            noise_kwargs = self.noise_kwargs()
            # run noise procedure
            self.daq.run("noise", file_name_kwargs, should_stop=self.should_stop, emit=self.emit,
                         indicators={"status_bar": self.status_bar}, **noise_kwargs)

    def shutdown(self):
        if self.z is not None:
            self.save()  # save data even if the procedure was aborted
        self.clean_up()  # delete references to data so that memory isn't hogged
        log.info("Finished sweep procedure")
     
    def fit_data(self):
        z = self.z - np.mean(self.z_offset, axis=1, keepdims=True)
        z[np.isnan(z)] = 0  # nans mess with the filter
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
        return z, frequencies, indices
        
    def compute_noise_bias(self):
        pass

    def save(self):
        self.status_bar.value = "Saving sweep data to file"
        file_path = os.path.join(self.directory, self.file_name())
        log.info("Saving data to %s", file_path)
        np.savez(file_path, freqs=self.freqs, z=self.z, z_offset=self.z_offset, f_offset=self.f_offset,
                 calibration=self.calibration, noise_bias=self.noise_bias, metadata=self.metadata)
                 
    def clean_up(self):
        self.status_bar.value = ""
        self.freqs = None
        self.z = None
        self.f_offset = None
        self.z_offset = None
        self.calibration = None
        self.noise_bias = None
        self.metadata = {"parameters": {}}
           
    @classmethod
    def load(cls, file_path):
        """
        Load the procedure output into a pymeasure Results class instance for the GUI.
        """
        # load in the data
        npz_file = load(file_path, allow_pickle=True)
        # create empty numpy structured array
        procedure = make_procedure_from_file(cls, npz_file)
        # make array with data
        results_dict = cls.make_results_dict(npz_file)
        # make a temporary file for the gui data
        results = make_results(results_dict, procedure)
        return results

    def calibrate(self):
        pass
    
    def get_sweep_data(self, index):
        raise NotImplementedError
        
    @classmethod
    def make_results_dict(cls, npz_file):
        raise NotImplementedError
        
    def noise_kwargs(self):
        raise NotImplementedError


class Sweep1(Sweep):
    # special parameters
    frequency = FloatParameter("Center Frequency", units="GHz", default=4.0)
    span = FloatParameter("Span", units="MHz", default=2)
    # gui data columns
    DATA_COLUMNS = ['f', 'i', 'q', 't', 'i_bias', 'q_bias', 'f_bias', 't_bias', 'i_psd', 'q_psd', 'f_psd']
    
    def startup(self):
        if self.should_stop():
            return
        self.status_bar.value = "Creating sweep data structures"
        self.setup_procedure_log(name='temperature', file_name='temperature.log')
        self.setup_procedure_log(name='resistance', file_name='resistance.log')
        self.setup_procedure_log(name='', file_name='procedure.log', filter=['resistance', 'temperature'])
        log.info("Starting sweep procedure")
        # create output data structures so that data is still saved after abort
        self.freqs = np.atleast_2d(np.linspace(self.frequency - self.span * 1e-3 / 2,
                                               self.frequency + self.span * 1e-3 / 2, self.n_points))
        if self.reverse_sweep:
            self.freqs = self.freqs[:, ::-1]                                       
        self.freqs = np.round(self.freqs, 9)  # round to nearest Hz        
        self.z = np.zeros(self.freqs.shape, dtype=np.complex64)
        # at least 0.1 MHz spacing
        self.f_offset = np.atleast_2d(np.linspace(self.freqs.min(), self.freqs.max(), int(max(3, 10 * self.span + 1))))
        self.z_offset = np.zeros(self.f_offset.shape, dtype=np.complex64)
        # save parameter metadata
        self.update_metadata()
    
    def get_sweep_data(self, index):
        z_offset = np.mean(self.z_offset)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            t = 20 * np.log10(np.abs(self.z[0, index] - z_offset)) - DB0
        t = np.nan if np.isinf(t) else t
        data = {"f": self.freqs[0, index],
                "i": self.z[0, index].real - z_offset.real,
                "q": self.z[0, index].imag - z_offset.imag,
                "t": t}
        return data
   
    @classmethod
    def make_results_dict(cls, npz_file):
        # get noise data
        try:
            noise_file = os.path.basename(npz_file.fid.name).split("_")
            noise_file[0] = "noise"
            noise_file = "_".join(noise_file)
            noise_file = os.path.join(os.path.dirname(npz_file.fid.name), noise_file)
            noise_npz_file = load(noise_file, allow_pickle=True)
            psd = noise_npz_file["psd"]
            freqs = noise_npz_file["f_psd"]
        except FileNotFoundError:
            psd = None
            freqs = None
        z_offset = np.mean(npz_file['z_offset'])
        # fill array
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            t = 20 * np.log10(np.abs(npz_file["z"][0, :] - z_offset)) - DB0
        t[np.isinf(t)] = np.nan

        result_dict = {"f": npz_file["freqs"][0, :],
                       "i": npz_file["z"][0, :].real - z_offset.real,
                       "q": npz_file["z"][0, :].imag - z_offset.imag,
                       "t": t}
        if psd is not None and freqs is not None:
            result_dict.update({"i_psd": psd[0, 0, 1:]['I'],
                                "q_psd": psd[0, 0, 1:]['Q'],
                                "f_psd": freqs[0, 1:]})
        if npz_file["noise_bias"].any():
            result_dict.update({"f_bias": npz_file["noise_bias"][0],
                                "t_bias": 20 * np.log10(np.abs(npz_file["noise_bias"][1] +
                                                               1j * npz_file["noise_bias"][2])),
                                "i_bias": npz_file["noise_bias"][1],
                                "q_bias": npz_file["noise_bias"][2]})
        return result_dict
        
    def compute_noise_bias(self):
        z, frequencies, indices = self.fit_data()
        
        self.noise_bias = np.array([frequencies[0],
                                    np.mean(z[0, indices[0]: indices[0] + 2].real),
                                    np.mean(z[0, indices[0]: indices[0] + 2].imag)])
                                    
        self.emit("results", {'f_bias': self.noise_bias[0],
                              't_bias': 20 * np.log10(np.abs(self.noise_bias[1] + 1j * self.noise_bias[2])),
                              'i_bias': self.noise_bias[1],
                              'q_bias': self.noise_bias[2]})
        
    def noise_kwargs(self):
        kwargs = {'directory': self.directory,
                  'attenuation': self.attenuation,
                  'sample_rate': self.sample_rate,
                  'total_atten': self.total_atten,
                  'frequency': self.frequency,
                  'time': self.noise[1],
                  'n_integrations': self.noise[2],
                  'off_res': bool(self.noise[3]),
                  'offset': self.noise[4],
                  'n_offset': self.noise[5]}
        return kwargs
        

class Sweep2(Sweep):
    # special parameters
    frequency1 = FloatParameter("Ch 1 Frequency", units="GHz", default=4.0)
    span1 = FloatParameter("Ch 1 Span", units="MHz", default=2)
    frequency2 = FloatParameter("Ch 2 Frequency", units="GHz", default=4.0)
    span2 = FloatParameter("Ch 2 Span", units="MHz", default=2)
    # gui data columns
    DATA_COLUMNS = ['f1', 'i1', 'q1', 't1', 'f1_bias', 't1_bias', 'i1_bias', 'q1_bias', 'i1_psd', 'q1_psd', 'f1_psd',
                    'f2', 'i2', 'q2', 't2', 'f2_bias', 't2_bias', 'i2_bias', 'q2_bias', 'i2_psd', 'q2_psd', 'f2_psd']

    def startup(self):
        if self.should_stop():
            return
        self.status_bar.value = "Creating sweep data structures"
        self.setup_procedure_log(name='temperature', file_name='temperature.log')
        self.setup_procedure_log(name='resistance', file_name='resistance.log')
        self.setup_procedure_log(name='', file_name='procedure.log', filter=['resistance', 'temperature'])
        log.info("Starting sweep procedure")
        # create output data structures so that data is still saved after abort
        self.freqs = np.vstack((np.linspace(self.frequency1 - self.span1 * 1e-3 / 2,
                                            self.frequency1 + self.span1 * 1e-3 / 2, self.n_points),
                                np.linspace(self.frequency2 - self.span2 * 1e-3 / 2,
                                            self.frequency2 + self.span2 * 1e-3 / 2, self.n_points)))
        if self.reverse_sweep:
            self.freqs = self.freqs[:, ::-1]
        self.freqs = np.round(self.freqs, 9)  # round to nearest Hz 
        self.z = np.zeros(self.freqs.shape, dtype=np.complex64)
        # at least 0.5 MHz spacing
        span = max(self.span1, self.span2)
        self.f_offset = np.vstack((np.linspace(self.freqs[0, :].min(), self.freqs[0, :].max(),
                                               int(max(3, 2 * span + 1))),
                                   np.linspace(self.freqs[1, :].min(), self.freqs[1, :].max(),
                                               int(max(3, 2 * span + 1)))))
        self.z_offset = np.zeros(self.f_offset.shape, dtype=np.complex64)
        self.calibration = np.zeros((2, 3, self.n_samples), dtype=[('I', np.float16), ('Q', np.float16)])
        self.noise_bias = np.zeros(6)
        # save parameter metadata
        self.update_metadata()

    def get_sweep_data(self, index):
        z_offset = np.mean(self.z_offset, axis=1)
        print(z_offset.shape)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            t1 = 20 * np.log10(np.abs(self.z[0, index] - z_offset[0])) - DB0
            t2 = 20 * np.log10(np.abs(self.z[1, index] - z_offset[1])) - DB0
        t1 = np.nan if np.isinf(t1) else t1
        t2 = np.nan if np.isinf(t2) else t2
        data = {"f1": self.freqs[0, index],
                "i1": self.z[0, index].real - z_offset[0].real,
                "q1": self.z[0, index].imag - z_offset[0].imag,
                "t1": t1,
                "f2": self.freqs[1, index],
                "i2": self.z[1, index].real - z_offset[1].real,
                "q2": self.z[1, index].imag - z_offset[1].imag,
                "t2": t2}
        return data
    
    @classmethod
    def make_results_dict(cls, npz_file):
        # get noise data
        try:
            noise_file = os.path.basename(npz_file.fid.name).split("_")
            noise_file[0] = "noise"
            noise_file = "_".join(noise_file)
            noise_file = os.path.join(os.path.dirname(npz_file.fid.name), noise_file)
            noise_npz_file = load(noise_file, allow_pickle=True)
            psd = noise_npz_file["psd"]
            freqs = noise_npz_file["f_psd"]
        except FileNotFoundError:
            psd = None
            freqs = None
        z_offset = np.mean(npz_file['z_offset'], axis=1)
        # fill array
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            t1 = 20 * np.log10(np.abs(npz_file["z"][0, :] - z_offset[0])) - DB0
            t2 = 20 * np.log10(np.abs(npz_file["z"][1, :] - z_offset[1])) - DB0
        t1[np.isinf(t1)] = np.nan
        t2[np.isinf(t2)] = np.nan

        result_dict = {"f1": npz_file["freqs"][0, :],
                       "i1": npz_file["z"][0, :].real - z_offset[0].real,
                       "q1": npz_file["z"][0, :].imag - z_offset[0].imag,
                       "t1": t1,
                       "f2": npz_file["freqs"][1, :],
                       "i2": npz_file["z"][1, :].real - z_offset[1].real,
                       "q2": npz_file["z"][1, :].imag - z_offset[1].imag,
                       "t2": t2}
        if psd is not None and freqs is not None:
            result_dict.update({"i1_psd": psd[0, 0, 1:]['I'],
                                "q1_psd": psd[0, 0, 1:]['Q'],
                                "f1_psd": freqs[0, 1:],
                                "i2_psd": psd[1, 0, 1:]['I'],
                                "q2_psd": psd[1, 0, 1:]['Q'],
                                "f2_psd": freqs[1, 1:]})
        if npz_file["noise_bias"].any():
            result_dict.update({"f1_bias": npz_file["noise_bias"][0],
                                "t1_bias": 20 * np.log10(np.abs(npz_file["noise_bias"][1] +
                                                                1j * npz_file["noise_bias"][2])),
                                "i1_bias": npz_file["noise_bias"][1],
                                "q1_bias": npz_file["noise_bias"][2],
                                "f2_bias": npz_file["noise_bias"][3],
                                "t2_bias": 20 * np.log10(np.abs(npz_file["noise_bias"][4] +
                                                                1j * npz_file["noise_bias"][5])),
                                "i2_bias": npz_file["noise_bias"][4],
                                "q2_bias": npz_file["noise_bias"][5]})
        return result_dict

    def compute_noise_bias(self):
        z, frequencies, indices = self.fit_data()

        self.noise_bias = np.array([frequencies[0],
                                    np.mean(z[0, indices[0]: indices[0] + 2].real),
                                    np.mean(z[0, indices[0]: indices[0] + 2].imag),
                                    frequencies[1],
                                    np.mean(z[1, indices[1]: indices[1] + 2].real),
                                    np.mean(z[1, indices[1]: indices[1] + 2].imag)])
                                    
        self.emit("results", {'f1_bias': self.noise_bias[0],
                              't1_bias': 20 * np.log10(np.abs(self.noise_bias[1] + 1j * self.noise_bias[2])),
                              'i1_bias': self.noise_bias[1],
                              'q1_bias': self.noise_bias[2],
                              'f2_bias': self.noise_bias[3],
                              't2_bias': 20 * np.log10(np.abs(self.noise_bias[4] + 1j * self.noise_bias[5])),
                              'i2_bias': self.noise_bias[4],
                              'q2_bias': self.noise_bias[5]})
        
    def noise_kwargs(self):        
        kwargs = {'directory': self.directory,
                  'attenuation': self.attenuation,
                  'sample_rate': self.sample_rate,
                  'total_atten': self.total_atten,
                  'frequency1': self.noise_bias[0],
                  'frequency2': self.noise_bias[3],
                  'time': self.noise[1],
                  'n_integrations': self.noise[2],
                  'off_res': bool(self.noise[3]),
                  'offset': self.noise[4],
                  'n_offset': self.noise[5]}
        return kwargs
        
    def calibrate(self):
        self.status_bar.value = "Calibrating IQ mixer phase and amplitude imbalance"
        # initialize in noise data mode
        self.daq.dac_atten.set_attenuation(self.attenuation)
        self.daq.adc.initialize(sample_rate=self.sample_rate * 1e6, n_samples=self.n_samples)
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

    directory = DirectoryParameter("Data Directory")
    attenuation = FloatParameter("DAC Attenuation", units="dB")
    sample_rate = FloatParameter("Sampling Rate", units="MHz", default=8e5)
    total_atten = IntegerParameter("Total Attenuation", units="dB", default=0)
    time = FloatParameter("Integration Time", default=1, units="s",)
    n_integrations = IntegerParameter("Number of Integrations", default=1)
    off_res = BooleanParameter("Take Off Resonance Data", default=True)
    offset = FloatParameter("Frequency Offset", units="MHz", default=-2)
    n_offset = IntegerParameter("# of Points", default=1)
    status_bar = Indicator("Status")

    def execute(self):
        if self.should_stop():
            log.warning(STOP_WARNING.format(self.__class__.__name__))
            return
        adc_atten = max(0, self.total_atten - self.attenuation)
        n_samples = int(self.time * self.sample_rate * 1e6)
        for index, _ in enumerate(self.freqs[0, :]):
            self.status_bar.value = "Taking noise data ({:d} / {:d})".format(index + 1, len(self.freqs[0, :]))
            # initialize the system in the right mode
            self.daq.initialize(self.freqs[:, index], dac_atten=self.attenuation, adc_atten=adc_atten,
                                sample_rate=self.sample_rate * 1e6, n_samples=n_samples)
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
        if self.noise is not None:
            self.save()  # save data even if the procedure was aborted
        self.clean_up()  # delete references to data so that memory isn't hogged
        log.info("Finished noise procedure")
        
    def save(self):
        self.status_bar.value = "Saving noise data to file"
        file_path = os.path.join(self.directory, self.file_name())
        log.info("Saving data to %s", file_path)
        np.savez(file_path, freqs=self.freqs, noise=self.noise, f_psd=self.f_psd, psd=self.psd, metadata=self.metadata)
                 
    def clean_up(self):
        self.status_bar.value = ""
        self.freqs = None
        self.noise = None
        self.f_psd = None
        self.psd = None
        self.metadata = {"parameters": {}}
                 
    def compute_psd(self):
        self.status_bar.value = "Computing PSDs"
        # n_points such that 1 kHz is the minimum possible freq
        n_points = min(self.noise.shape[-1], int(self.sample_rate * 1e6 / 1000))
        kwargs = {'nperseg': n_points, 'fs': self.sample_rate * 1e6, 'return_onesided': True,
                  'detrend': 'constant', 'scaling': 'density', 'axis': -1, 'window': 'hanning'}
        _, i_psd = sig.welch(self.noise['I'], **kwargs)
        _, q_psd = sig.welch(self.noise['Q'], **kwargs)
        # average multiple PSDs together
        i_psd = np.mean(i_psd, axis=-2)
        q_psd = np.mean(q_psd, axis=-2)
        # save in array
        self.psd['I'] = i_psd
        self.psd['Q'] = q_psd

    def get_noise_data(self):
        raise NotImplementedError


class Noise1(Noise):
    frequency = FloatParameter("Bias Frequency", units="GHz", default=4.0)

    def startup(self):
        if self.should_stop():
            return
        self.status_bar.value = "Creating noise data structures"
        self.setup_procedure_log(name='temperature', file_name='temperature.log')
        self.setup_procedure_log(name='resistance', file_name='resistance.log')
        self.setup_procedure_log(name='', file_name='procedure.log', filter=['resistance', 'temperature'])
        log.info("Starting noise procedure")
        # create output data structures so that data is still saved after abort
        n_noise = int(1 + self.off_res * self.n_offset)
        n_points = int(self.time * self.sample_rate * 1e6)
        offset = np.linspace(0, self.offset, self.off_res * self.n_offset + 1) * 1e-3  # offset in MHz
        self.freqs = self.frequency + offset.reshape((1, offset.size))
        self.noise = np.zeros((1, n_noise, self.n_integrations, n_points), dtype=[('I', np.float16), ('Q', np.float16)])

        n_points = min(self.noise.shape[-1], int(self.sample_rate * 1e6 / 1000))
        fft_freq = np.fft.rfftfreq(n_points, d=1 / (self.sample_rate * 1e6))
        n_fft = fft_freq.size
        self.psd = np.zeros((1, n_noise, n_fft), dtype=[('I', np.float32), ('Q', np.float32)])
        self.f_psd = fft_freq.reshape((1, fft_freq.size))
        self.update_metadata()

    def get_noise_data(self):
        data = {"f_psd": self.f_psd[0, 1:],
                "i_psd": self.psd[0, 0, 1:]['I'],
                "q_psd": self.psd[0, 0, 1:]['Q']}
        return data


class Noise2(Noise):
    frequency1 = FloatParameter("Channel 1 Bias Frequency", units="GHz", default=4.0)
    frequency2 = FloatParameter("Channel 2 Bias Frequency", units="GHz", default=4.0)
    
    def startup(self):
        if self.should_stop():
            return
        self.status_bar.value = "Creating noise data structures"
        self.setup_procedure_log(name='temperature', file_name='temperature.log')
        self.setup_procedure_log(name='resistance', file_name='resistance.log')
        self.setup_procedure_log(name='', file_name='procedure.log', filter=['resistance', 'temperature'])
        log.info("Starting noise procedure")
        # create output data structures so that data is still saved after abort
        n_noise = int(1 + self.off_res * self.n_offset)
        n_points = int(self.time * self.sample_rate * 1e6)
        offset = np.linspace(0, self.offset, self.off_res * self.n_offset + 1) * 1e-3  # offset in MHz
        self.freqs = np.array([self.frequency1 + offset, self.frequency2 + offset])
        self.noise = np.zeros((2, n_noise, self.n_integrations, n_points), dtype=[('I', np.float16), ('Q', np.float16)])

        n_points = min(self.noise.shape[-1], int(self.sample_rate * 1e6 / 1000))
        fft_freq = np.fft.rfftfreq(n_points, d=1 / (self.sample_rate * 1e6))
        n_fft = fft_freq.size
        self.psd = np.zeros((2, n_noise, n_fft), dtype=[('I', np.float32), ('Q', np.float32)])
        self.f_psd = np.array([fft_freq, fft_freq])
        self.update_metadata()

    def get_noise_data(self):
        data = {"f1_psd": self.f_psd[0, 1:],
                "i1_psd": self.psd[0, 0, 1:]['I'],
                "q1_psd": self.psd[0, 0, 1:]['Q'],
                "f2_psd": self.f_psd[1, 1:],
                "i2_psd": self.psd[1, 0, 1:]['I'],
                "q2_psd": self.psd[1, 0, 1:]['Q']}
        return data


class Pulse(MKIDProcedure):
    # outputs
    freqs = None
    pulses = None
    offset = None

    directory = DirectoryParameter("Data Directory")
    sweep_file = FileParameter("Sweep File")
    attenuation = FloatParameter("DAC Attenuation", units="dB")
    sample_rate = FloatParameter("Sampling Rate", units="MHz", default=0.8)
    
    sigma = FloatParameter("N Sigma Trigger", default=4)
    integration_time = FloatParameter("Time Per Integration", units="s", default=1)
    dead_time = FloatParameter("Dead Time", units="s", default=1000e-6)
    total_atten = IntegerParameter("Total Attenuation", units="dB", default=0)
    n_pulses = IntegerParameter("Number of Pulses", default=10000)
    n_trace = IntegerParameter("Data Points per Pulses", default=2000)
    noise = VectorParameter("Noise", default=[1, 1, 10], ui_class=NoiseInput)
    status_bar = Indicator("Status")
    
    count_rates = []

    def execute(self):
        if self.should_stop():
            log.warning(STOP_WARNING.format(self.__class__.__name__))
            return
        # take noise data
        if self.noise[0]:
            self.status_bar.value = "Taking noise data"
            # get file name kwargs from file_name
            file_name_kwargs = self.file_name_parts()
            file_name_kwargs["prefix"] = "noise"
            # run noise procedure
            self.daq.run("noise", file_name_kwargs, should_stop=self.should_stop, emit=self.emit,
                         indicators={"status_bar": self.status_bar}, **self.noise_kwargs())

        adc_atten = max(0, self.total_atten - self.attenuation)
        n_samples = int(self.integration_time * self.sample_rate * 1e6)
        self.status_bar.value = "Calibrating IQ mixer offset"

        # take zero point (laser gets turned off here if noise wasn't run)
        with warnings.catch_warnings():
            # ignoring warnings for setting infinite attenuation
            warnings.simplefilter("ignore", UserWarning)
            self.daq.initialize(self.freqs, dac_atten=np.inf, adc_atten=adc_atten,
                                sample_rate=self.sample_rate * 1e6, n_samples=n_samples)
        zero = self.daq.adc.take_iq_point()
        self.offset['I'] = zero.real
        self.offset['Q'] = zero.imag

        # initialize the system in the right mode
        self.status_bar.value = "Computing noise level"
        self.daq.dac_atten.initialize(self.attenuation)
        self.daq.adc.initialize(sample_rate=self.sample_rate * 1e6, n_samples=n_samples, n_trace=self.n_trace)

        data = self.daq.adc.take_noise_data(1)
        sigma = np.zeros(data.shape[0], dtype=[('I', np.float64), ('Q', np.float64)])
        sigma['I'] = median_abs_deviation(data['I'].astype(np.float64), scale='normal', axis=-1).ravel()
        sigma['Q'] = median_abs_deviation(data['Q'].astype(np.float64), scale='normal', axis=-1).ravel()

        # take the data
        self.status_bar.value = "Taking pulse data"
        self.daq.laser.set_state(self.laser)
        n_pulses = 0
        amplitudes = np.zeros((data.shape[0], self.n_pulses))
        while n_pulses < self.n_pulses:
            # channel, n_pulses, n_trace ['I' or 'Q']
            data, triggers = self.daq.adc.take_pulse_data(sigma, n_sigma=self.sigma, dead_time=self.dead_time)

            new_pulses = data.shape[1]
            space_left = self.n_pulses - n_pulses

            if isinstance(self.pulses, np.memmap):  # reload the mem-map so that we don't keep all the pulses in RAM
                self.pulses = np.lib.format.open_memmap(self.pulses.filename, mode="r+")
            self.pulses[:, n_pulses: new_pulses + n_pulses, :]['I'] = data[:, :space_left, :]['I']
            self.pulses[:, n_pulses: new_pulses + n_pulses, :]['Q'] = data[:, :space_left, :]['Q']
            if isinstance(self.pulses, np.memmap):  # ensure that the data is flushed to disk
                self.pulses.flush()

            responses_i = data[:, :space_left, :]['I'] - np.median(data[:, :space_left, :]['I'], axis=2, keepdims=True)
            responses_q = data[:, :space_left, :]['Q'] - np.median(data[:, :space_left, :]['Q'], axis=2, keepdims=True)
            amplitudes[:, n_pulses:n_pulses + new_pulses] = np.max(np.sqrt(responses_i**2 + responses_q**2), axis=2)

            n_pulses += new_pulses
            self.emit("progress", n_pulses / self.n_pulses * 100)

            for index, count_rate in enumerate(self.count_rates):
                count_rate.value = np.sum(triggers[index, :]) / (n_samples / (self.sample_rate * 1e6))

            pulses = self.get_pulse_data(data, triggers)
            self.emit('results', pulses, clear=True)
            self.emit_amplitude_data(amplitudes, new_pulses)

            if self.should_stop():
                log.warning(STOP_WARNING.format(self.__class__.__name__))
                return
        # record system state after data taking
        self.metadata.update(self.daq.system_state())

    def shutdown(self):
        # zero out count rate indicators
        for index, count_rate in enumerate(self.count_rates):
            count_rate.value = 0
        self.daq.laser.set_state(self.daq.laser.OFF_STATE)  # turn laser off
        if self.pulses is not None:
            self.save()  # save data even if the procedure was aborted
        self.clean_up()  # delete references to data so that memory isn't hogged
        log.info("Finished noise procedure")

    def save(self):
        self.status_bar.value = "Saving pulse data to file"
        file_path = os.path.join(self.directory, self.file_name())
        log.info("Saving data to %s", file_path)
        if isinstance(self.pulses, np.memmap):
            np.savez(file_path, freqs=self.freqs, pulses=self.pulses.filename, zero=self.offset, metadata=self.metadata)
        else:
            np.savez(file_path, freqs=self.freqs, pulses=self.pulses, zero=self.offset, metadata=self.metadata)

    def clean_up(self):
        self.status_bar.value = ""
        self.freqs = None
        self.pulses = None
        self.offset = None
        self.metadata = {"parameters": {}}
        
    @classmethod
    def load(cls, file_path):
        """
        Load the procedure output into a pymeasure Results class instance for the GUI.
        """
        # load in the data
        npz_file = load(file_path, allow_pickle=True)
        # create empty numpy structured array
        procedure = make_procedure_from_file(cls, npz_file)
        # make array with data
        results_dict = cls.make_results_dict(npz_file)
        # make a temporary file for the gui data
        results = make_results(results_dict, procedure)
        return results
        
    def noise_kwargs(self):
        raise NotImplementedError
        
        
class Pulse1(Pulse):
    frequency = FloatParameter("Bias Frequency", units="GHz", default=4.0)
    count_rate = FloatIndicator("Count Rate", units="Hz", default=0)
    count_rates = [count_rate]
    ui = BooleanListInput.set_labels(["254 nm", "406.6 nm", "671 nm", "808 nm", "920 nm", "980 nm", "1120 nm",
                                      "1310 nm"])  # class factory
    laser = VectorParameter("Laser", default=[0, 0, 0, 0, 0, 0, 0, 0], length=8, ui_class=ui)
    DATA_COLUMNS = ["i", "q", "i_loop", "q_loop", 'i_psd', 'q_psd', 'f_psd', 'hist_x', 'hist_y']
    
    def startup(self):
        if self.should_stop():
            return
        self.status_bar.value = "Loading sweep data"
        result = Sweep1.load(self.sweep_file)
        self.emit("results", {'i_loop': result.data['i'],
                              'q_loop': result.data['q']})
        
        self.status_bar.value = "Creating pulse data structures"
        self.setup_procedure_log(name='temperature', file_name='temperature.log')
        self.setup_procedure_log(name='resistance', file_name='resistance.log')
        self.setup_procedure_log(name='', file_name='procedure.log', filter=['resistance', 'temperature'])
        log.info("Starting pulse procedure")
        # create output data structures so that data is still saved after abort
        self.freqs = np.array([self.frequency])
        # use a memmap so that large amounts of data can be taken
        file_path = os.path.splitext(os.path.join(self.directory, self.file_name()))[0] + ".npy"
        self.pulses = np.lib.format.open_memmap(file_path, mode="w+", shape=(1, self.n_pulses, self.n_trace),
                                                dtype=[('I', np.float16), ('Q', np.float16)])
        self.offset = np.zeros(1, dtype=[('I', np.float32), ('Q', np.float32)])
        self.update_metadata() 
        
    def get_pulse_data(self, pulses, triggers):
        data = {}
        try:
            data["i"] = pulses['I'][0, np.argmax(triggers[0, :]), :] - self.offset['I'][0]
            data["q"] = pulses['Q'][0, np.argmax(triggers[0, :]), :] - self.offset['Q'][0]
        except ValueError:  # attempt to get argmax of an empty sequence
            pass
        return data

    def emit_amplitude_data(self, amplitudes, new_pulses):
        counts, bins = np.histogram(amplitudes[amplitudes != 0], bins='auto', density=True)
        data = {'hist_x': bins, 'hist_y': counts}
        self.emit("results", data, clear=True)
        
    def noise_kwargs(self):
        kwargs = {'directory': self.directory,
                  'attenuation': self.attenuation,
                  'sample_rate': self.sample_rate,
                  'total_atten': self.total_atten,
                  'frequency': self.frequency,
                  'time': self.noise[1],
                  'n_integrations': self.noise[2],
                  'off_res': False,
                  'offset': 0,
                  'n_offset': 0}
        return kwargs

    @classmethod
    def make_results_dict(cls, npz_file):
        # get noise data
        try:
            noise_file = os.path.basename(npz_file.fid.name).split("_")
            noise_file[0] = "noise"
            noise_file = "_".join(noise_file)
            noise_file = os.path.join(os.path.dirname(npz_file.fid.name), noise_file)
            noise_npz_file = load(noise_file, allow_pickle=True)
            psd = noise_npz_file["psd"]
            freqs = noise_npz_file["f_psd"]
        except FileNotFoundError:
            psd = None
            freqs = None
        result = Sweep1.load(npz_file['metadata'].item()['parameters']['sweep_file'])
        responses = np.sqrt(npz_file['pulses']['I']**2 + npz_file['pulses']['Q']**2)
        amplitudes = (np.max(responses, axis=2) - np.median(responses, axis=2))
        bins, counts = np.histogram(amplitudes[amplitudes != 0], bins='auto')
        result_dict = {'i_loop': result.data['i'],
                       'q_loop': result.data['q'],
                       'i': npz_file['pulses']['I'][0, 0, :] - npz_file['offset']["I"][0],
                       'q': npz_file['pulses']['Q'][0, 0, :] - npz_file['offset']["Q"][0],
                       'hist_x': bins,
                       'hist_y': counts}
        if psd is not None and freqs is not None:
            result_dict.update({"i_psd": psd[0, 0, 1:]['I'],
                                "q_psd": psd[0, 0, 1:]['Q'],
                                "f_psd": freqs[0, 1:]})
        return result_dict


class Pulse2(Pulse):
    frequency1 = FloatParameter("Channel 1 Bias Frequency", units="GHz", default=4.0)
    frequency2 = FloatParameter("Channel 2 Bias Frequency", units="GHz", default=4.0)
    
    count_rate1 = FloatIndicator("Channel 1 Count Rate", units="Hz", default=0)
    count_rate2 = FloatIndicator("Channel 2 Count Rate", units="Hz", default=0)
    count_rates = [count_rate1, count_rate2]
    ui = BooleanListInput.set_labels(["808 nm", "920 nm", "980 nm", "1120 nm", "1310 nm"])  # class factory
    laser = VectorParameter("Laser", default=[0, 0, 0, 0, 0], length=5, ui_class=ui)

    DATA_COLUMNS = ["i1", "q1", "i2", "q2", "i1_loop", "q1_loop", "i2_loop", "q2_loop", 'i1_psd', 'q1_psd', 'f1_psd',
                    'i2_psd', 'q2_psd', 'f2_psd', 'peaks1', 'peaks2']

    def startup(self):
        if self.should_stop():
            return
        self.status_bar.value = "Loading sweep data"
        result = Sweep2.load(self.sweep_file)
        self.emit("results", {'i1_loop': result.data['i1'],
                              'q1_loop': result.data['q1'],
                              'i2_loop': result.data['i2'],
                              'q2_loop': result.data['q2']})
        
        self.status_bar.value = "Creating pulse data structures"
        self.setup_procedure_log(name='temperature', file_name='temperature.log')
        self.setup_procedure_log(name='resistance', file_name='resistance.log')
        self.setup_procedure_log(name='', file_name='procedure.log', filter=['resistance', 'temperature'])
        log.info("Starting pulse procedure")
        # create output data structures so that data is still saved after abort
        self.freqs = np.array([self.frequency1, self.frequency2])
        # use a memmap so that large amounts of data can be taken
        file_path = os.path.splitext(os.path.join(self.directory, self.file_name()))[0] + ".npy"
        self.pulses = np.lib.format.open_memmap(file_path, mode="w+", shape=(2, self.n_pulses, self.n_trace),
                                                dtype=[('I', np.float16), ('Q', np.float16)])
        self.offset = np.zeros(2, dtype=[('I', np.float32), ('Q', np.float32)])
        self.update_metadata()
        
    def get_pulse_data(self, pulses, triggers):
        data = {}
        try:
            data["i1"] = pulses['I'][0, np.argmax(triggers[0, :]), :] - self.offset['I'][0]
            data["q1"] = pulses['Q'][0, np.argmax(triggers[0, :]), :] - self.offset['Q'][0]
        except ValueError:  # attempt to get argmax of an empty sequence
            pass
        try:
            data["i2"] = pulses['I'][1, np.argmax(triggers[1, :]), :] - self.offset['I'][1]
            data["q2"] = pulses['Q'][1, np.argmax(triggers[1, :]), :] - self.offset['Q'][1]
        except ValueError:  # attempt to get argmax of an empty sequence
            pass
        return data

    def emit_amplitude_data(self, amplitudes, new_pulses):
        data = {'peaks1': amplitudes[0, -new_pulses:], 'peaks2': amplitudes[1, -new_pulses:]}
        self.emit("results", data)
        
    def noise_kwargs(self):
        kwargs = {'directory': self.directory,
                  'attenuation': self.attenuation,
                  'sample_rate': self.sample_rate,
                  'total_atten': self.total_atten,
                  'frequency1': self.frequency1,
                  'frequency2': self.frequency2,
                  'time': self.noise[1],
                  'n_integrations': self.noise[2],
                  'off_res': False,
                  'offset': 0,
                  'n_offset': 0}
        return kwargs

    @classmethod
    def make_results_dict(cls, npz_file):
        # get noise data
        try:
            noise_file = os.path.basename(npz_file.fid.name).split("_")
            noise_file[0] = "noise"
            noise_file = "_".join(noise_file)
            noise_file = os.path.join(os.path.dirname(npz_file.fid.name), noise_file)
            noise_npz_file = load(noise_file, allow_pickle=True)
            psd = noise_npz_file["psd"]
            freqs = noise_npz_file["f_psd"]
        except FileNotFoundError:
            psd = None
            freqs = None
        responses = np.sqrt(npz_file['pulses']['I']**2 + npz_file['pulses']['Q']**2)
        amplitudes = (np.max(responses, axis=2) - np.median(responses, axis=2))
        result = Sweep2.load(npz_file['metadata'].item()['parameters']['sweep_file'])
        result_dict = {'i1_loop': result.data['i1'],
                       'q1_loop': result.data['q1'],
                       'i1': npz_file['pulses']['I'][0, 0, :] - npz_file['offset']["I"][0],
                       'q1': npz_file['pulses']['Q'][0, 0, :] - npz_file['offset']["Q"][0],
                       'i2_loop': result.data['i2'],
                       'q2_loop': result.data['q2'],
                       'i2': npz_file['pulses']['I'][1, 0, :] - npz_file['offset']["I"][1],
                       'q2': npz_file['pulses']['Q'][1, 0, :] - npz_file['offset']["Q"][1],
                       'peaks1': amplitudes[0],
                       'peaks2': amplitudes[1]}
        if psd is not None and freqs is not None:
            result_dict.update({"i1_psd": psd[0, 0, 1:]['I'],
                                "q1_psd": psd[0, 0, 1:]['Q'],
                                "f1_psd": freqs[0, 1:],
                                "i2_psd": psd[1, 0, 1:]['I'],
                                "q2_psd": psd[1, 0, 1:]['Q'],
                                "f2_psd": freqs[1, 1:]})
        return result_dict


class Fit(FitProcedure):
    FIT_PARAMETERS = ["f0", "qi", "qc", "xa", "a", "gain0", "gain1", "gain2", "phase0", "phase1", "phase2",
                      "alpha", "beta", "gamma", "delta"]
    TOOLTIPS = {"f0": "the low power resonance frequency",
                "qi": "the internal quality factor",
                "qc": "the coupling quality factor",
                "xa": ("the fractional resonator skew (as defined in "
                       "Zobrist et al. 2020 doi:10.1117/1.JATIS.7.1.010501)"),
                "a": ("the inductive nonlinearity (the formula for 'a' can be found in "
                      "Swenson et al. 2013 doi:10.1063/1.4794808)"),
                "gain0": "0th term in the gain polynomial",
                "gain1": "1st term in the gain polynomial",
                "gain2": "2nd term in the gain polynomial",
                "phase0": "0th term in the phase polynomial",
                "phase1": "1st term in the phase polynomial",
                "phase2": ("2nd term in the phase polynomial (there are few physical motivations for this parameter "
                           "so it is usually not fit)"),
                "alpha": "IQ mixer amplitude imbalance (ideal is 1)",
                "beta": "IQ mixer phase imbalance (ideal is 0 or pi)",
                "gamma": ("IQ mixer Q offset (this parameter is determined automatically during the procedure and "
                          "doesn't need to be fit)"),
                "delta": ("IQ mixer Q offset (this parameter is determined automatically during the procedure and "
                          "doesn't need to be fit)")}
    DERIVED_PARAMETERS = ["q0", "tau", "fr", "fm"]
    ranges = None
    do_fit = None

    def execute(self):
        for i, channel in enumerate(self.CHANNELS):
            if self.should_stop():
                log.warning(STOP_WARNING.format(self.__class__.__name__))
                return
            if self.do_fit is not None and not self.do_fit[i]:
                continue

            # Do the fit.
            log.info(f"Processing channel {channel}.")
            # Try to fit the whole resonator by using multiple sweeps.
            if self.config_file is not None:
                log.info(f"Loading sweep from {self.config_file}.")
                sweep = mc.Sweep.from_file(self.config_file, unique=False, sort=False)  # load the sweep
                # Find the right resonator and loop in the sweep.
                loop, resonator = self.find_loop(sweep, channel)
                log.info(f"Fitting loop {loop.name}.")

                # We've already fit the resonator, so grab the loop from the cache.
                if resonator.name in self.fitted_resonators.keys():
                    log.info(f"Resonator {resonator.name} has been pre-fit. Using this solution.")
                    r, guesses = self.fitted_resonators[resonator.name]
                    for ii, l in enumerate(r.loops):
                        if l.name == loop.name:
                            loop = l  # switch out the loop for the prefitted version
                            guess = guesses[ii]

                # We haven't already fit the resonator, so fit it.
                else:
                    log.info(f"Fitting resonator {resonator.name}.")
                    guesses = []
                    for l in resonator.loops:
                        self.mask_loop(l, i)
                        guesses.append(self.guess(l))
                        if l.name == loop.name:
                            guess = guesses[-1]

                    # Only include the nonlinear fit if we are varying that parameter.
                    if self.a[0]:
                        extra_fits = (mc.experiments.temperature_fit, mc.experiments.power_fit,
                                      mc.experiments.nonlinear_fit, mc.experiments.linear_fit)
                    # Don't do the linear fit if we are fixing a non-zero nonlinearity.
                    elif not self.a[0] and not self.a[1]:  # fixed at non-zero nonlinearity
                        extra_fits = (mc.experiments.temperature_fit, mc.experiments.power_fit)
                    else:
                        extra_fits = (mc.experiments.temperature_fit, mc.experiments.power_fit,
                                      mc.experiments.linear_fit)
                    n_fits = len(resonator.loops) * (len(extra_fits) * 2 + 1)

                    class Progress():
                        def __init__(self):
                            self.index = 0

                        def __call__(s):
                            s.index += 1
                            self.emit("progress", 100 * (i + s.index / n_fits) / len(self.CHANNELS))

                    mc.experiments.multiple_fit(resonator, extra_fits=extra_fits, guess=guesses, fit_type='lmfit',
                                                callback=Progress(), iterations=2)
                    self.fitted_resonators[resonator.name] = (resonator, guesses)
                    log.info(f"lmfit report for {loop.name}: {loop.lmfit_results['best']['label']}:\n"
                             + loop.fit_report(label='best', fit_type='lmfit', return_string=True))

            # Just fit the loop if we don't have the whole resonator.
            else:
                loop = mc.Loop.from_file(self.sweep_file, channel=channel - 1)
                log.info(f"Fitting loop {loop.name}.")
                self.mask_loop(loop, i)
                guess = self.guess(loop)
                mc.experiments.basic_fit(loop, guess=guess, fit_type='lmfit')
                log.info(f"lmfit report for {loop.name}: {loop.lmfit_results['best']['label']}:\n"
                         + loop.fit_report(label='best', fit_type='lmfit', return_string=True))
            result = loop.lmfit_results['best']['result']

            # Emit the results to GUI.
            results_dict = {param + f"_{channel}": get_fit_result(result, param)
                            for param in self.FIT_PARAMETERS}
            results_dict.update({param + f"_{channel}": get_fit_result(result, param)
                                 for param in self.DERIVED_PARAMETERS})
            keys = ["i{}_guess", "i{}_fit", "q{}_guess", "q{}_fit", "f{}_guess", "f{}_fit", "t{}_guess", "t{}_fit"]
            keys = [key.format(channel) for key in keys]
            f = np.linspace(loop.f[loop.mask].min(), loop.f[loop.mask].max(), 10 * loop.f[loop.mask].size)
            z_guess = mc.models.S21.model(guess, f) - loop.offset_calibration.mean()
            z_fit = mc.models.S21.model(result.params, f) - loop.offset_calibration.mean()
            values = [z_guess.real, z_fit.real, z_guess.imag, z_fit.imag, f, f,
                      20 * np.log10(np.abs(z_guess)) - DB0, 20 * np.log10(np.abs(z_fit)) - DB0]
            results_dict.update({key: value for key, value in zip(keys, values)})
            results_dict.update({"filename": self.file_name(), "sweep_file": os.path.basename(self.sweep_file),
                                 f"channel{channel}": channel})
            self.emit("results", results_dict)

            # Save the results.
            log.info(f"Saving channel {channel}.")
            with open(os.path.join(self.directory, self.file_name()), "r") as f:
                config = yaml.load(f, Loader=yaml.Loader)
            config['guess'] = config.get('guess', {})
            config['result'] = config.get('result', {})
            config['guess'].update({param + f"_{channel}": float(guess[param].value) for param in self.FIT_PARAMETERS})
            config['guess'].update({param + f"_{channel}": float(guess[param].value)
                                    for param in self.DERIVED_PARAMETERS})
            config['result'].update({param + f"_{channel}": get_fit_result(result, param)
                                     for param in self.FIT_PARAMETERS})
            config['result'].update({param + f"_{channel}": get_fit_result(result, param)
                                     for param in self.DERIVED_PARAMETERS})
            with open(os.path.join(self.directory, self.file_name()), "w") as f:
                yaml.dump(config, f)
            self.emit("progress", 100 * (i + 1) / len(self.CHANNELS))


    def shutdown(self):
        if self.clear_fits:
            self.fitted_resonators.clear()
        log.info("Finished fit procedure")

    def guess(self, loop):
        # Create the guess.
        # The 'nonlinear_resonance' and 'quadratic_phase' options change the default guess.
        log.info(f"Creating the guess for {loop.name}.")
        guess = mc.models.S21.guess(loop.z[loop.mask], loop.f[loop.mask],
                                    imbalance=loop.imbalance_calibration,
                                    offset=loop.offset_calibration,
                                    nonlinear_resonance=True if self.a[0] else False,
                                    quadratic_phase=True if self.phase2[0] else False)

        # Overload the guess with GUI options if specified.
        for param in self.FIT_PARAMETERS:
            options = getattr(self, param)
            if param == "a":  # "a_sqrt" is really being varied not "a"
                param = "a_sqrt"
                for index, option in enumerate(options[1:]):
                    if not np.isnan(option) and not np.isinf(option):
                        options[index + 1] = np.sqrt(option)
            guess[param].set(vary=bool(options[0]))  # vary
            if not np.isnan(options[1]):  # value
                guess[param].set(value=float(options[1]))
            if not np.isnan(options[2]):  # min
                guess[param].set(min=float(options[2]))
            if not np.isnan(options[3]):  # max
                guess[param].set(max=float(options[3]))

        return guess

    def find_loop(self, sweep, channel):
        for resonator in sweep.resonators:
            for loop in resonator.loops:
                if os.path.basename(self.sweep_file) in loop.name and f"'channel': {channel - 1}" in loop.name:
                    break
            else:
                continue  # only executed if the inner for loop did not break
            break  # only executed if the inner for loop did break
        else:
            # only exected if the for loops did not break
            raise ValueError(f"Sweep {self.config_file} does not contain Loop "
                             f"{os.path.basename(self.sweep_file)} and channel {channel - 1}")
        return loop, resonator

    def mask_loop(self, loop, i):
        # Mask the data.
        if self.ranges is not None:
            f_min = loop.f.min()
            f_max = loop.f.max()
            f_med = np.median(loop.f)
            lower = self.ranges[2 * i] / 100 * (f_med - f_min) + f_min
            upper = (1 - self.ranges[2 * i + 1] / 100) * (f_max - f_med) + f_med
            loop.mask_from_bounds(lower=lower, upper=upper)

    @classmethod
    def load(cls, file_path):
        """
        Load the procedure output into a pymeasure Results class instance for the GUI.
        """
        # load in the data
        with open(file_path, "r") as f:
            config = yaml.load(f, Loader=yaml.Loader)
        # make a procedure object with the right parameters
        procedure = cls()
        for name, value in config['parameters'].items():
            setattr(procedure, name, value)
        procedure.refresh_parameters()  # Enforce update of meta data
        results_dict = cls.make_results_dict(config)
        results_dict['filename'] = os.path.basename(file_path)
        # make a temporary file for the gui data
        results = make_results(results_dict, procedure)
        return results


class Fit1(Fit):
    CHANNELS = [1]
    DATA_COLUMNS = ['i1_loop', 'i1_guess', 'i1_fit', 'q1_loop', 'q1_guess', 'q1_fit', 'f1', 'f1_guess', 'f1_fit',
                    't1', 't1_guess', 't1_fit', 'channel1', "sweep_file", "filename"]
    DATA_COLUMNS.extend([param + f"_{channel}" for channel in CHANNELS for param in Fit.FIT_PARAMETERS])
    DATA_COLUMNS.extend([param + f"_{channel}" for channel in CHANNELS for param in Fit.DERIVED_PARAMETERS])
    ranges = VectorParameter("Trim frequency ranges", default=[0, 0], length=2, ui_class=RangeInput)

    def startup(self):
        if self.should_stop():
            return
        result = Sweep1.load(self.sweep_file)
        self.emit("results", {'i1_loop': result.data['i'],
                              'q1_loop': result.data['q'],
                              'f1': result.data['f'],
                              't1': result.data['t']})
        self.setup_procedure_log(name='', file_name='procedure.log', filter=['resistance', 'temperature'])
        log.info("Starting fit procedure")

    @classmethod
    def make_results_dict(cls, config):
        sweep_result = Sweep1.load(config['parameters']['sweep_file'])
        result_dict = {'i1_loop': sweep_result.data['i'],
                       'q1_loop': sweep_result.data['q'],
                       'f1': sweep_result.data['f'],
                       't1': sweep_result.data['t'],
                       'channel1': 1,
                       "sweep_file": os.path.basename(config['parameters']['sweep_file'])}
        result_dict.update(config['result'])
        keys = ["i1_guess", "i1_fit", "q1_guess", "q1_fit", "f1_guess", "f1_fit", "t1_guess", "t1_fit"]
        f = np.array(sweep_result.data["f"])
        f_min = f.min()
        f_max = f.max()
        f_med = np.median(f)
        f_range = config['parameters']['ranges']
        mask = np.ones_like(f, dtype=bool)
        if not np.isnan(f_range[0]):
            mask = mask & (f >= f_range[0] / 100 * (f_med - f_min) + f_min)
        if not np.isnan(f_range[1]):
            mask = mask & (f <= (1 - f_range[1] / 100) * (f_max - f_med) + f_med)
        f = f[mask]
        f = np.linspace(f.min(), f.max(), 10 * f.size)
        # create lmfit parameters objects
        guess = lm.Parameters()
        guess.add_many([[key[:-2], value] for key, value in config['guess'].items() if key.endswith(f"_1")])
        result = lm.Parameters()
        result.add_many([[key[:-2], value[0] if isinstance(value, (list, tuple)) else value]
                         for key, value in config['result'].items() if key.endswith(f"_1")])
        # evaluate the model
        z_guess = mc.models.S21.model(guess, f) - (config['guess']['gamma_1']
                                                   + 1j * config['guess']['delta_1'])
        z_fit = mc.models.S21.model(result, f) - (np.atleast_1d(config['result']['gamma_1'])[0]
                                                  + 1j * np.atleast_1d(config['result']['delta_1'])[0])
        values = [z_guess.real, z_fit.real, z_guess.imag, z_fit.imag, f, f,
                  20 * np.log10(np.abs(z_guess)) - DB0, 20 * np.log10(np.abs(z_fit)) - DB0]
        result_dict.update({key: value for key, value in zip(keys, values)})
        return result_dict


class Fit2(Fit):
    CHANNELS = [1, 2]
    DATA_COLUMNS = ['i1_loop', 'i1_guess', 'i1_fit', 'q1_loop', 'q1_guess', 'q1_fit', 'f1', 'f1_guess', 'f1_fit',
                    't1', 't1_guess', 't1_fit', 'channel1',
                    'i2_loop', 'i2_guess', 'i2_fit', 'q2_loop', 'q2_guess', 'q2_fit', 'f2', 'f2_guess', 'f2_fit',
                    't2', 't2_guess', 't2_fit', 'channel2', "sweep_file", "filename"]
    DATA_COLUMNS.extend([param + f"_{channel}" for channel in CHANNELS for param in Fit.FIT_PARAMETERS])
    DATA_COLUMNS.extend([param + f"_{channel}" for channel in CHANNELS for param in Fit.DERIVED_PARAMETERS])
    ui = RangeInput.set_labels(["Channel 1:", "Channel 2:"])  # class factory
    ranges = VectorParameter("Trim frequency ranges", default=[0, 0, 0, 0], length=4, ui_class=ui)
    ui = BooleanListInput.set_labels(["Fit Channel 1", "Fit Channel 2"])  # class factory
    do_fit = VectorParameter("", default=[1, 1], length=2, ui_class=ui)

    def startup(self):
        if self.should_stop():
            return
        result = Sweep2.load(self.sweep_file)
        self.emit("results", {'i1_loop': result.data['i1'],
                              'q1_loop': result.data['q1'],
                              'i2_loop': result.data['i2'],
                              'q2_loop': result.data['q2'],
                              'f1': result.data['f1'],
                              't1': result.data['t1'],
                              'f2': result.data['f2'],
                              't2': result.data['t2']})
        self.setup_procedure_log(name='', file_name='procedure.log', filter=['resistance', 'temperature'])
        log.info("Starting fit procedure")

    @classmethod
    def make_results_dict(cls, config):
        sweep_result = Sweep2.load(config['parameters']['sweep_file'])
        result_dict = {'i1_loop': sweep_result.data['i1'],
                       'q1_loop': sweep_result.data['q1'],
                       'i2_loop': sweep_result.data['i2'],
                       'q2_loop': sweep_result.data['q2'],
                       'f1': sweep_result.data['f1'],
                       't1': sweep_result.data['t1'],
                       'f2': sweep_result.data['f2'],
                       't2': sweep_result.data['t2'],
                       'channel1': 1,
                       'channel2': 2,
                       "sweep_file": os.path.basename(config['parameters']['sweep_file'])}
        result_dict.update(config['result'])
        for i in range(1, 3):
            if not config['parameters']['do_fit'][i - 1]:
                continue
            keys = ["i{}_guess", "i{}_fit", "q{}_guess", "q{}_fit", "f{}_guess", "f{}_fit", "t{}_guess", "t{}_fit"]
            keys = [key.format(i) for key in keys]
            f = np.array(sweep_result.data[f'f{i}'])
            f_min = f.min()
            f_max = f.max()
            f_med = np.median(f)
            f_range = config['parameters']['ranges']
            mask = np.ones_like(f, dtype=bool)
            if not np.isnan(f_range[2 * i - 2]):
                mask = mask & (f >= f_range[2 * i - 2] / 100 * (f_med - f_min) + f_min)
            if not np.isnan(f_range[2 * i - 1]):
                mask = mask & (f <= (1 - f_range[2 * i - 1] / 100) * (f_max - f_med) + f_med)
            f = f[mask]
            f = np.linspace(f.min(), f.max(), 10 * f.size)
            # create lmfit parameters objects
            guess = lm.Parameters()
            guess.add_many(*[[key[:-2], value] for key, value in config['guess'].items() if key.endswith(f"_{i}")])
            result = lm.Parameters()
            result.add_many(*[[key[:-2], value[0] if isinstance(value, (list, tuple)) else value]
                              for key, value in config['result'].items() if key.endswith(f"_{i}")])
            # evaluate the model
            z_guess = mc.models.S21.model(guess, f) - (config['guess'][f'gamma_{i}']
                                                       + 1j * config['guess'][f'delta_{i}'])
            z_fit = mc.models.S21.model(result, f) - (np.atleast_1d(config['result'][f'gamma_{i}'])[0]
                                                      + 1j * np.atleast_1d(config['result'][f'delta_{i}'])[0])
            values = [z_guess.real, z_fit.real, z_guess.imag, z_fit.imag, f, f,
                      20 * np.log10(np.abs(z_guess)) - DB0, 20 * np.log10(np.abs(z_fit)) - DB0]
            result_dict.update({key: value for key, value in zip(keys, values)})
        return result_dict
