import PyDAQmx
import numpy as np
from time import sleep
from scipy.ndimage.filters import median_filter

DAQmx_Val_Rising = PyDAQmx.DAQmxConstants.DAQmx_Val_Rising
DAQmx_Val_FiniteSamps = PyDAQmx.DAQmxConstants.DAQmx_Val_FiniteSamps
DAQmx_Val_Cfg_Default = PyDAQmx.DAQmxConstants.DAQmx_Val_Cfg_Default
DAQmx_Val_Volts = PyDAQmx.DAQmxConstants.DAQmx_Val_Volts
DAQmx_Val_DC = PyDAQmx.DAQmxConstants.DAQmx_Val_DC
DAQmx_Val_GroupByChannel = PyDAQmx.DAQmxConstants.DAQmx_Val_GroupByChannel


class NI6120:
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
        # collect more data than requested at a time to trigger on pulses
        self.trigger_factor = 20

    def initialize(self, application, channels=None, sample_rate=None, num_samples=None):
        self.reset()        
        if application == "pulse_data":
            # collect more data than requested at a time to trigger on pulses
            if num_samples is None:
                num_samples = self.trigger_factor * self.samples_per_channel
            else:
                num_samples = self.trigger_factor * num_samples
        elif application == "noise_data":
            pass
        elif application == "sweep_data":
            pass
        self._create_channels(channels=channels)
        self._set_channel_coupling()
        self._disable_aa_filter()
        self._configure_sampling(sample_rate=sample_rate, num_samples=num_samples)

        
    def take_noise_data(self, n_triggers):
        data = []
        n_channels = len(self.channels)
        for _ in range(n_channels):
            data.append(np.zeros((n_triggers, int(self.samples_per_channel))))
        for index in range(n_triggers):
            rand_time = np.random.random_sample() * 0.001  # no longer than a millisecond
            sleep(rand_time)
            sample = self._acquire_readings()
            for channel_index in range(n_channels):
                data[channel_index][index, :] = sample[channel_index, :]
        return tuple(data)
        
    def take_pulse_data(self, n_triggers, n_sigma=4):
        # initialize data array
        n_channels = len(self.channels)
        n_samples = int(self.samples_per_channel / self.trigger_factor)
        data = np.zeros((n_channels, n_triggers, n_samples))
        # compute triggers
        sigmas = self._find_sigma_levels(search_length=500)
        # collect data
        n_pulses = 0
        while n_pulses < n_triggers:
            sample = self._acquire_readings().T 
            sample -= np.median(sample, axis=1)
            # time ordered pulse indices
            time_indices, _ = np.where(np.abs(sample) > n_sigma * sigmas)
            # enforce one trigger per n_samples
            logic = np.ones(time_indices.shape, dtype=bool)            
            for index, time_index in enumerate(time_indices):
                if np.any(time_index - time_indices[:index] < n_samples):
                    logic[index] = False
            # enforce no triggers at the beginning or end of sample
            logic = np.logical_and(logic, time_indices > n_samples / 2) 
            logic = np.logical_and(
                logic, time_indices < self.samples_per_channel - n_samples / 2)        
            # trim triggers and convert to index array
            time_indices = time_indices[logic]
            index_array = (np.ones((len(time_indices), n_samples)) *
                           np.arange(-n_samples // 2, n_samples // 2) +
                           np.atleast_2d(time_indices).T)
            # add triggers to data array
            new_data = sample[index_array]  # (triggers, samples, channel)
            new_data = np.transpose(new_data, (2, 0, 1))  # (channel, triggers, samples)
            n_new_pulses = new_data.shape[1]
            if n_new_pulses + n_pulses > n_triggers:
                n_new_pulses = n_triggers - n_pulses
            data[:, n_pulses:n_pulses + n_new_pulses, :] = new_data[:, :n_new_pulses, :] 
            # update counter
            n_pulses += n_new_pulses
        return tuple(data)           
            
    
    def take_iq_point(self):
        channel_data = self._acquire_readings()
        # combine I and Q signals
        data = np.zeros(int(len(channel_data) / 2))
        for index in range(int(len(channel_data) / 2)):
            data[index] = (np.median(channel_data[2 * index]) +
                           1j * np.median(channel_data[2 * index + 1]))
        return data

    def reset(self):
        """
        Resets the digitizer.
        """
        error = PyDAQmx.DAQmxResetDevice(self.device)
        return error

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
        error = False
        for channel in self.channels:
            error = self.session.CreateAIVoltageChan(channel, "", DAQmx_Val_Cfg_Default,
                                                     self.input_range_min,
                                                     self.input_range_max,
                                                     DAQmx_Val_Volts, None) or error
        return error

    def _configure_sampling(self, sample_rate=None, num_samples=None):
        """
        Configures the sampling.
        """
        if sample_rate is not None:
            self.sample_rate = sample_rate

        if num_samples is not None:
            self.samples_per_channel = num_samples

        error = self.session.CfgSampClkTiming("", np.int(self.sample_rate),
                                              DAQmx_Val_Rising, DAQmx_Val_FiniteSamps,
                                              np.int(self.samples_per_channel))
        return error

    def _set_channel_coupling(self):
        error = False
        for channel in self.channels:
            error = self.session.SetAICoupling(channel, DAQmx_Val_DC) or error
        return error

    def _acquire_readings(self):
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
        data = np.array(np.hsplit(data, n_channels))

        return data

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
