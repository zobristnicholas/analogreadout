import os
import logging
import numpy as np
from time import sleep
from scipy.ndimage.filters import median_filter
try:
    import PyDAQmx
    DAQmx_Val_Rising = PyDAQmx.DAQmxConstants.DAQmx_Val_Rising
    DAQmx_Val_FiniteSamps = PyDAQmx.DAQmxConstants.DAQmx_Val_FiniteSamps
    DAQmx_Val_Cfg_Default = PyDAQmx.DAQmxConstants.DAQmx_Val_Cfg_Default
    DAQmx_Val_Volts = PyDAQmx.DAQmxConstants.DAQmx_Val_Volts
    DAQmx_Val_DC = PyDAQmx.DAQmxConstants.DAQmx_Val_DC
    DAQmx_Val_GroupByChannel = PyDAQmx.DAQmxConstants.DAQmx_Val_GroupByChannel
except (NotImplementedError, ImportError):
    pass  # allow for import if PyDAQmx is not configured
try:
    import matlab.engine
except ImportError:
    pass  # allow for import if matlab is not installed

log = logging.getLogger(__name__)
log.addHandler(logging.NullHandler())


class DigitizerABC:
    def acquire_readings(self):
        raise NotImplementedError
    
    def take_noise_data(self, n_triggers):
        n_channels = len(self.channels) // 2
        data = np.empty((n_channels, n_triggers, int(self.samples_per_channel)),
                        dtype=[('I', np.float16), ('Q', np.float16)])
        for index in range(n_triggers):
            rand_time = np.random.random_sample() * 0.001  # no longer than a millisecond
            sleep(rand_time)
            sample = self.acquire_readings()
            data[:, index, :]['I'] = sample[::2, :]
            data[:, index, :]['Q'] = sample[1::2, :]
        return data
        
    def take_pulse_data(self, trigger_level, n_sigma=4):
        # initialize data array
        n_channels = len(self.channels) // 2
        n_samples = int(self.samples_per_partition)
        # collect data
        sample = self.acquire_readings()
        # turn the trigger level into a sigma
        sigma = np.empty((sample.shape[0], 1))
        sigma[::2, 0] = trigger_level['I']
        sigma[1::2, 0] = trigger_level['Q']
        # time ordered pulse indices
        logic = np.abs(sample - np.median(sample, axis=-1, keepdims=True)) > n_sigma * sigma
        triggered = np.nonzero(logic.any(axis=0))[0]
        # enforce one trigger per n_samples
        for index, trigger in enumerate(triggered):
            previous_triggers = np.any(trigger - triggered[:index] < n_samples)
            beginning_trigger = trigger < n_samples // 2
            ending_trigger = trigger > self.samples_per_channel - n_samples // 2
            if previous_triggers or beginning_trigger or ending_trigger:
                logic[:, trigger] = False
        # create array of trigger channels
        trigger_indices = np.nonzero(logic.any(axis=0))[0]
        n_triggers = int(trigger_indices.size)
        triggers = np.empty((n_channels, n_triggers), dtype=bool)
        # put triggers in dataset
        data = np.empty((n_channels, n_triggers, n_samples), dtype=[('I', np.float16), ('Q', np.float16)])
        for ii, time_index in enumerate(trigger_indices):
            triggers[:, ii] = logic[:, time_index].reshape((-1, 2)).any(axis=1)
            data[:, ii, :]['I'] = sample[::2, time_index - n_samples // 2: time_index + n_samples // 2 + n_samples % 2]
            data[:, ii, :]['Q'] = sample[1::2, time_index - n_samples // 2: time_index + n_samples // 2 + n_samples % 2]
        return data, triggers

    def take_iq_point(self):
        channel_data = self.acquire_readings()
        # combine I and Q signals
        data = np.zeros(int(len(channel_data) / 2), dtype=np.complex64)
        for index in range(int(len(channel_data) / 2)):
            data[index] = (np.mean(channel_data[2 * index]) + 1j * np.mean(channel_data[2 * index + 1]))
        return data


class NI6120(DigitizerABC):
    def __init__(self):
        self.session = PyDAQmx.Task()
        self.device = "Dev1"
        # set timeout value
        self.timeout = 20.0  # seconds
        self.read = PyDAQmx.int32()  # for sampsPerChanRead
        # set default input range, can be 42, 20, 10, 5, 2, 1, 0.5, 0.2 Volts
        self.input_range_max = 0.2
        self.input_range_min = -1.0 * self.input_range_max
        # set default physical channel(s) to use
        self.channels = ["/ai1", "/ai0", "/ai3", "/ai2"]  # I1, Q1, I2, Q2
        self.channels = [self.device + channel for channel in self.channels]
        # set default sample rate
        self.sample_rate = 8.0e5  # samples per second
        # set default number of samples per channel
        self.samples_per_channel = 2e4
        # set trace / collection ratio for pulse data
        self.samples_per_partition = 2000
        # collect more data than requested at a time to trigger on pulses
        log.info("Connected to: National Instruments PCI-6120")

    def initialize(self, channels=None, sample_rate=None, n_samples=None, n_trace=None):
        self.reset()
        self._create_channels(channels=channels)
        self._set_channel_coupling()
        self._disable_aa_filter()
        self._configure_sampling(sample_rate=sample_rate, n_samples=n_samples, n_trace=n_trace)

    def reset(self):
        """
        Resets the digitizer.
        """
        error = PyDAQmx.DAQmxResetDevice(self.device)
        self.session = PyDAQmx.Task()
        return error
    
    def acquire_readings(self):
        """
        Returns the digitized readings for two channels. An array is returned of shape
        (self.channels.size, self.samples_per_channel) and dtype =  np.float64
        """
        n_channels = len(self.channels)
        size = np.int(self.samples_per_channel * n_channels)
        data = np.empty((size,), dtype=np.float64)

        self.session.StartTask()
        # byref() Returns a pointer lookalike to a C instance
        samples_per_channel_read = PyDAQmx.byref(self.read)
        # as opposed to DAQmx_Val_GroupByScanNumber
        fill_mode = DAQmx_Val_GroupByChannel

        self.session.ReadAnalogF64(np.int(self.samples_per_channel), self.timeout,
                                   fill_mode, data, size, samples_per_channel_read, None)
        self.session.StopTask()

        # get data
        data = data.reshape((n_channels, -1))

        return data

    def _find_sigma_levels(self, search_length=50):
        # define some sizes
        n_kernel = int(self.samples_per_channel / self.trigger_factor)
        n_kernel = n_kernel // 2 * 2 + 1  # round up to odd number for kernel size
        n_samples = self.samples_per_channel - (n_kernel - 1)
        n_channels = len(self.channels)
        boundary = (n_kernel - 1) / 2
        # compute the median absolute deviation for a bunch of acquisitions
        mad_data = np.ones((n_channels, n_samples * search_length)) * np.inf
        for index in range(search_length):
            data = self._acquire_readings()
            s = slice(index * n_samples, (index + 1) * n_samples)
            mad_data[s] = median_filter(np.abs(data - np.median(data, axis=1)),
                                        (1, n_kernel))[:, boundary:-boundary]
        # the index with the smallest median absolute deviation has the least
        # pulse contamination
        indices = np.argmin(mad_data)
        sigmas = 1.4826 * mad_data[range(mad_data.shape[0]), indices]
        return sigmas
        
    def _create_channels(self, channels=None):
        """
        Creates four channels by default.
        """
        if channels is not None:
            self.channels = channels
        message = "must have an equal number of I and Q channels"
        assert len(self.channels) % 2 == 0, message
        error = False
        for channel in self.channels:
            error = self.session.CreateAIVoltageChan(channel, "", DAQmx_Val_Cfg_Default,
                                                     self.input_range_min,
                                                     self.input_range_max,
                                                     DAQmx_Val_Volts, None) or error
        return error

    def _configure_sampling(self, sample_rate=None, n_samples=None, n_trace=None):
        """
        Configures the sampling.
        """
        if sample_rate is not None:
            self.sample_rate = sample_rate

        if n_samples is not None:
            self.samples_per_channel = n_samples
            
        if n_trace is not None:
            self.samples_per_partition = n_trace

        error = self.session.CfgSampClkTiming("", np.int(self.sample_rate),
                                              DAQmx_Val_Rising, DAQmx_Val_FiniteSamps,
                                              np.int(self.samples_per_channel))
        return error

    def _set_channel_coupling(self):
        error = False
        for channel in self.channels:
            error = self.session.SetAICoupling(channel, DAQmx_Val_DC) or error
        return error

    def _enable_aa_filter(self):
        """
        Enable the built-in 5-pole Bessel 100kHz anti-alias filter of the NI6120.
        """
        error = self.session.SetAILowpassEnable("", 1)
        return error

    def _disable_aa_filter(self):
        """
        Disable the built-in 5-pole Bessel 100kHz anti-alias filter of the NI6120.
        """
        error = self.session.SetAILowpassEnable("", 0)
        return error

    def _calibrate(self):
        """
        Calibrate the digitizer.
        """
        error = self.session.DAQmxSelfCal(self.device)
        return error


class Advantech1840(DigitizerABC):
    def __init__(self):
        self.session = matlab.engine.start_matlab()
        matlab_path = os.path.join(os.path.abspath(os.path.dirname(__file__)), "matlab")
        self.session.addpath(matlab_path, nargout=0)
        # set default input range, can be 10, 5, 2, 1, 0.5, 0.2, 0.1 Volts
        self.input_range_max = 0.1
        self.input_range_min = -1.0 * self.input_range_max
        # set default physical channel(s) to use
        self.n_channels = [0, 1]  # I1, Q1
        # set default sample rate
        self.sample_rate = 2.5e6  # samples per second
        # set default number of samples per channel
        self.samples_per_channel = 2.5e6
        # set trace / collection ratio for pulse data
        self.samples_per_partition = 2**13
        # collect more data than requested at a time to trigger on pulses
        log.info("Connected to: Advantech PCIe-1840")
    
    def initialize(self, channels=None, sample_rate=None, n_samples=None, n_trace=None):
        self.reset()
        if channels is not None:
            self.channels = channels
        if sample_rate is not None:
            self.sample_rate = sample_rate
        if n_samples is not None:
            self.samples_per_channel = n_samples
        if n_trace is not None:
            self.samples_per_partition = n_trace
        n_channels = len(self.channels)
        sample_rate, n_samples, error = self.session.advantech_1840_startup(
                n_channels, self.sample_rate, self.samples_per_channel, nargout=3)

        self.sample_rate = sample_rate  # could be set even if error
        self.samples_per_channel = n_samples
        if error:
            raise RuntimeError(error)
            
    def take_iq_point(self):
        if self.samples_per_channel > 10000:
            return super().take_iq_point()
        else:
            n_channels = len(self.channels)
            sample, error = self.session.advantech_1840_instant(
                    n_channels, self.samples_per_channel, nargout=2)
            if error:
                raise RuntimeError(error)
            sample = self._matlab_to_numpy(sample)
            data = np.empty(n_channels // 2, dtype=np.complex64)
            for index in range(n_channels // 2):
                data[index] = np.mean(sample[2 * index::n_channels] + 1j * sample[2 * index + 1::n_channels])
            return data
    
    def acquire_readings(self):
        sample, error = self.session.advantech_1840_acquire(nargout=2)
        if error:
            raise RuntimeError(error)
        sample = self._matlab_to_numpy(sample)
        data = sample.reshape((-1, len(self.channels))).T
        return data   
    
    def reset(self):
        pass
    
    @staticmethod
    def _matlab_to_numpy(array):
        # np.array(sample) is slow so we access the internal list for the conversion
        return np.array(array._data).reshape(array.size, order='F').T
