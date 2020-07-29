import os
import logging
import tempfile
import warnings
import numpy as np
import scipy.signal as sig
from scipy.interpolate import interp1d
# from scipy.stats import median_abs_deviation  # TODO: uncomment when we can use a modern version of python
from mkidplotter import (SweepBaseProcedure, MKIDProcedure, NoiseInput, Results, DirectoryParameter, BooleanListInput,
                         Indicator, FloatIndicator, FileParameter)
from pymeasure.experiment import (IntegerParameter, FloatParameter, BooleanParameter,
                                  VectorParameter)
from analogreadout.utils import load

log = logging.getLogger(__name__)
log.addHandler(logging.NullHandler())

STOP_WARNING = "Caught the stop flag in the '{}' procedure"


def median_abs_deviation(x, scale=1.0):  # stand in for scipy.stats median_abs_deviation which doesn't exist until version 1.5
    if isinstance(scale, str):
        if scale.lower() == 'normal':
            scale = 0.6744897501960817  # special.ndtri(0.75)
    else:
        raise ValueError("{} is not a valid scale value.".format(scale))
    med = np.median(x, axis=None)
    mad = np.median(np.abs(x - med))
    return mad / scale

  
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
    z_offset_interp = None

    sample_rate = FloatParameter("Sample Rate", units="MHz", default=0.8)
    n_samples = IntegerParameter("Samples to Average", default=20000)
    n_points = IntegerParameter("Number of Points", default=500)
    total_atten = FloatParameter("Total Attenuation", units="dB", default=0)
    reverse_sweep = BooleanParameter("Reverse Sweep Direction", default=False)
    wait_temp_min = IntegerParameter("Set Temperature Minimum Wait Time", units="minutes", default=0)
    wait_temp_max = IntegerParameter("Set Temperature Maximum Wait Time", units="minutes", default=0)
    noise = VectorParameter("Noise", length=6, default=[1, 1, 10, 1, -2, 1], ui_class=NoiseInput)
    status_bar = Indicator("Status")

    def execute(self):
        if self.should_stop():
            log.warning(STOP_WARNING.format(self.__class__.__name__))
            return
        # TODO: set_field when there's an instrument hooked up
        self.daq.thermometer.set_temperature(self.temperature, min_wait=self.wait_temp_min, max_wait=self.wait_temp_max)
        # calibrate the data (if possible)
        self.calibrate()
        # initialize the system in the right mode
        with warnings.catch_warnings():
            # ignoring warnings for setting infinite attenuation
            warnings.simplefilter("ignore", UserWarning)
            self.daq.initialize(self.freqs[:, 0], dac_atten=np.inf, adc_atten=np.inf,
                                sample_rate=self.sample_rate * 1e6, n_samples=25 * self.n_samples)
        # loop through the frequencies and take data
        self.status_bar.value = "Calibrating IQ mixer offset"
        for index, _ in enumerate(self.f_offset[0, :]):
            self.daq.dac.set_frequency(self.f_offset[:, index])
            if index == 0:
                self.daq.adc.take_iq_point()  # first data point is sometimes garbage
            self.z_offset[:, index] = self.daq.adc.take_iq_point()
            self.emit('progress', index / (self.f_offset.shape[1] + self.freqs.shape[1]) * 100)
            log.debug("taking zero index: %d", index)
            if self.should_stop():
                log.warning(STOP_WARNING.format(self.__class__.__name__))
                return
        self.compute_offset()
        # initialize the system in the right mode
        self.status_bar.value = "Sweeping"
        adc_atten = max(0, self.total_atten - self.attenuation)
        self.daq.initialize(self.freqs[:, 0], dac_atten=self.attenuation, adc_atten=adc_atten,
                            sample_rate=self.sample_rate * 1e6, n_samples=self.n_samples)
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
        
    @staticmethod
    def interpolate_offset(freqs, f_offset, z_offset):
        z_offset_interp = np.zeros(freqs.shape, dtype=np.complex64)
        for ind in range(f_offset.shape[0]):
            z_offset_interp[ind, :] = interp1d(f_offset[ind, :], z_offset[ind, :])(freqs[ind, :])
        return z_offset_interp
        
    def compute_offset(self):
        self.z_offset_interp = self.interpolate_offset(self.freqs, self.f_offset, self.z_offset)
     
    def fit_data(self):
        z = self.z - self.z_offset_interp
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
        self.z_offset_interp = None
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
        self.setup_procedure_log(name=__name__, file_name='procedure.log')
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
        db0 = 10 * np.log10(1e-3 / 50)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            t = 20 * np.log10(np.abs(self.z[0, index] - self.z_offset_interp[0, index])) - db0
        t = np.nan if np.isinf(t) else t
        data = {"f": self.freqs[0, index],
                "i": self.z[0, index].real - self.z_offset_interp[0, index].real,
                "q": self.z[0, index].imag - self.z_offset_interp[0, index].imag,
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
        z_offset_interp = cls.interpolate_offset(npz_file["freqs"], npz_file["f_offset"], npz_file["z_offset"])
        # fill array
        db0 = 10 * np.log10(1e-3 / 50)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            t = 20 * np.log10(np.abs(npz_file["z"][0, :] - z_offset_interp[0, :])) - db0
        t[np.isinf(t)] = np.nan

        result_dict = {"f": npz_file["freqs"][0, :],
                       "i": npz_file["z"][0, :].real - z_offset_interp[0, :].real,
                       "q": npz_file["z"][0, :].imag - z_offset_interp[0, :].imag,
                       "t": t}
        if psd is not None and freqs is not None:
            result_dict.update({"i_psd": psd[0, 0, :]['I'],
                                "q_psd": psd[0, 0, :]['Q'],
                                "f_psd": freqs[0, :]})
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
        self.setup_procedure_log(name=__name__, file_name='procedure.log')
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
        # at least 0.1 MHz spacing
        span = max(self.span1, self.span2)
        self.f_offset = np.vstack((np.linspace(self.freqs[0, :].min(), self.freqs[0, :].max(),
                                               int(max(3, 10 * span + 1))),
                                   np.linspace(self.freqs[1, :].min(), self.freqs[1, :].max(),
                                               int(max(3, 10 * span + 1)))))
        self.z_offset = np.zeros(self.f_offset.shape, dtype=np.complex64)
        self.calibration = np.zeros((2, 3, self.n_samples), dtype=[('I', np.float16), ('Q', np.float16)])
        self.noise_bias = np.zeros(6)
        # save parameter metadata
        self.update_metadata()

    def get_sweep_data(self, index):
        db0 = 10 * np.log10(1e-3 / 50)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            t1 = 20 * np.log10(np.abs(self.z[0, index] - self.z_offset_interp[0, index])) - db0
            t2 = 20 * np.log10(np.abs(self.z[1, index] - self.z_offset_interp[1, index])) - db0
        t1 = np.nan if np.isinf(t1) else t1
        t2 = np.nan if np.isinf(t2) else t2    
        data = {"f1": self.freqs[0, index],
                "i1": self.z[0, index].real - self.z_offset_interp[0, index].real,
                "q1": self.z[0, index].imag - self.z_offset_interp[0, index].imag,
                "t1": t1,
                "f2": self.freqs[1, index],
                "i2": self.z[1, index].real - self.z_offset_interp[1, index].real,
                "q2": self.z[1, index].imag - self.z_offset_interp[1, index].imag,
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
        z_offset_interp = cls.interpolate_offset(npz_file["freqs"], npz_file["f_offset"], npz_file["z_offset"])
        # fill array
        db0 = 10 * np.log10(1e-3 / 50)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            t1 = 20 * np.log10(np.abs(npz_file["z"][0, :] - z_offset_interp[0, :])) - db0
            t2 = 20 * np.log10(np.abs(npz_file["z"][1, :] - z_offset_interp[1, :])) - db0
        t1[np.isinf(t1)] = np.nan
        t2[np.isinf(t2)] = np.nan

        result_dict = {"f1": npz_file["freqs"][0, :],
                       "i1": npz_file["z"][0, :].real - z_offset_interp[0, :].real,
                       "q1": npz_file["z"][0, :].imag - z_offset_interp[0, :].imag,
                       "t1": t1,
                       "f2": npz_file["freqs"][1, :],
                       "i2": npz_file["z"][1, :].real - z_offset_interp[1, :].real,
                       "q2": npz_file["z"][1, :].imag - z_offset_interp[1, :].imag,
                       "t2": t2}
        if psd is not None and freqs is not None:
            result_dict.update({"i1_psd": psd[0, 0, :]['I'],
                                "q1_psd": psd[0, 0, :]['Q'],
                                "f1_psd": freqs[0, :],
                                "i2_psd": psd[1, 0, :]['I'],
                                "q2_psd": psd[1, 0, :]['Q'],
                                "f2_psd": freqs[1, :]})
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
        adc_atten = max(0, self.total_atten - self.attenuation)
        self.daq.initialize(self.freqs[:, 0], dac_atten=self.attenuation, adc_atten=adc_atten,
                            sample_rate=self.sample_rate * 1e6, n_samples=self.n_samples)
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
        # n_points such that 100 Hz is the minimum possible freq
        n_points = min(self.noise.shape[-1], int(self.sample_rate * 1e6 / 100))
        kwargs = {'nperseg': n_points, 'fs': self.sample_rate * 1e6, 'return_onesided': True,
                  'detrend': 'constant', 'scaling': 'density', 'axis': -1, 'window': 'hanning'}
        _, i_psd = sig.welch(self.noise['I'], **kwargs)
        _, q_psd = sig.welch(self.noise['Q'], **kwargs)
        # average multiple PSDs together
        i_psd = np.mean(i_psd, axis=-2)
        q_psd = np.mean(q_psd, axis=-2)
        # fix zero point
        i_psd[:, :, 0] = i_psd[:, :, 1]
        q_psd[:, :, 0] = q_psd[:, :, 1]
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
        self.setup_procedure_log(name=__name__, file_name='procedure.log')
        log.info("Starting noise procedure")
        # create output data structures so that data is still saved after abort
        n_noise = int(1 + self.off_res * self.n_offset)
        n_points = int(self.time * self.sample_rate * 1e6)
        offset = np.linspace(0, self.offset, self.off_res * self.n_offset + 1) * 1e-3  # offset in MHz
        self.freqs = self.frequency + offset.reshape((1, offset.size))
        self.noise = np.zeros((1, n_noise, self.n_integrations, n_points), dtype=[('I', np.float16), ('Q', np.float16)])

        n_points = min(self.noise.shape[-1], int(self.sample_rate * 1e6 / 100))
        fft_freq = np.fft.rfftfreq(n_points, d=1 / (self.sample_rate * 1e6))
        n_fft = fft_freq.size
        self.psd = np.zeros((1, n_noise, n_fft), dtype=[('I', np.float32), ('Q', np.float32)])
        self.f_psd = fft_freq.reshape((1, fft_freq.size))
        self.update_metadata()

    def get_noise_data(self):
        data = {"f_psd": self.f_psd[0, :],
                "i_psd": self.psd[0, 0, :]['I'],
                "q_psd": self.psd[0, 0, :]['Q']}
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
        self.setup_procedure_log(name=__name__, file_name='procedure.log')
        log.info("Starting noise procedure")
        # create output data structures so that data is still saved after abort
        n_noise = int(1 + self.off_res * self.n_offset)
        n_points = int(self.time * self.sample_rate * 1e6)
        offset = np.linspace(0, self.offset, self.off_res * self.n_offset + 1) * 1e-3  # offset in MHz
        self.freqs = np.array([self.frequency1 + offset, self.frequency2 + offset])
        self.noise = np.zeros((2, n_noise, self.n_integrations, n_points), dtype=[('I', np.float16), ('Q', np.float16)])

        n_points = min(self.noise.shape[-1], int(self.sample_rate * 1e6 / 100))
        fft_freq = np.fft.rfftfreq(n_points, d=1 / (self.sample_rate * 1e6))
        n_fft = fft_freq.size
        self.psd = np.zeros((2, n_noise, n_fft), dtype=[('I', np.float32), ('Q', np.float32)])
        self.f_psd = np.array([fft_freq, fft_freq])
        self.update_metadata()

    def get_noise_data(self):
        data = {"f1_psd": self.f_psd[0, :],
                "i1_psd": self.psd[0, 0, :]['I'],
                "q1_psd": self.psd[0, 0, :]['Q'],
                "f2_psd": self.f_psd[1, :],
                "i2_psd": self.psd[1, 0, :]['I'],
                "q2_psd": self.psd[1, 0, :]['Q']}
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
    dead_time = FloatParameter("Dead Time", units="s", default=100e-6)
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

        # take zero point
        with warnings.catch_warnings():
            # ignoring warnings for setting infinite attenuation
            warnings.simplefilter("ignore", UserWarning)
            self.daq.initialize(self.freqs, dac_atten=np.inf, adc_atten=np.inf,
                                sample_rate=self.sample_rate * 1e6, n_samples=n_samples)
        zero = self.daq.adc.take_iq_point()
        self.offset['I'] = zero.real
        self.offset['Q'] = zero.imag

        # initialize the system in the right mode (laser off)
        self.status_bar.value = "Computing noise level"
        self.daq.initialize(self.freqs, dac_atten=self.attenuation, adc_atten=adc_atten,
                            sample_rate=self.sample_rate * 1e6, n_samples=n_samples, n_trace=self.n_trace)
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
        self.setup_procedure_log(name=__name__, file_name='procedure.log')
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
        bins, counts = np.histogram(amplitudes[amplitudes != 0], bins='auto')
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
        result = Sweep1.load(cls.sweep_file)

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
            result_dict.update({"i_psd": psd[0, 0, :]['I'],
                                "q_psd": psd[0, 0, :]['Q'],
                                "f_psd": freqs[0, :]})
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
        self.setup_procedure_log(name=__name__, file_name='procedure.log')
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
        result = Sweep2.load(cls.sweep_file)
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
            result_dict.update({"i1_psd": psd[0, 0, :]['I'],
                                "q1_psd": psd[0, 0, :]['Q'],
                                "f1_psd": freqs[0, :],
                                "i2_psd": psd[1, 0, :]['I'],
                                "q2_psd": psd[1, 0, :]['Q'],
                                "f2_psd": freqs[1, :]})
        return result_dict
