import PyDAQmx
import numpy as np
from time import sleep

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
        self.channels = ["/ai0", "/ai1", "/ai2", "/ai3"]
        self.channels = [self.device + channel for channel in self.channels]
        # set default sample rate
        self.sample_rate = 8.0e5  # samples per second
        # set default number of samples per channel
        self.samples_per_channel = 2e4

    def initialize(self, application, channels=None, sample_rate=None, num_samples=None):
        if application == "pulse_data":
            pass

        elif application == "noise_data":
            self._create_channels(channels=channels)
            self._configure_sampling(sample_rate=sample_rate, num_samples=num_samples)
            self._set_channel_coupling()
            self._disable_aa_filter()

        elif application == "sweep_data":
            pass
        
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

    def reset(self):
        """
        Resets the digitizer.
        """
        error = PyDAQmx.DAQmxResetDevice(self.device)
        return error

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
        data = np.array(np.hsplit(data, n_channels)) * (2**15-1) / self.input_range_max

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
