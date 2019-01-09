import visa
import logging
import numpy as np
from time import sleep

log = logging.getLogger(__name__)
log.addHandler(logging.NullHandler())


class AgilentMSO6054A:
    def __init__(self, address, channels=(1, 3)):
        # recast channels as integers
        channels = [int(channel) for channel in channels]
        self.channels = channels
        # set default number of samples per channel
        self.samples_per_channel = 1e3
        try:
            resource_manager = visa.ResourceManager()
        except Exception:
            resource_manager = visa.ResourceManager('@py')
        self.session = resource_manager.open_resource(address)
        identity = self._query_ascii_values("*IDN?", 's')
        identity = [s.strip() for s in identity]
        log.info("Connected to: %s %s, s/n: %s, version: %s", *identity)

    def initialize(self, application):
        self.reset()
        self._write("*CLS")

        # set each channel at 50 Ohm impedance
        self._write(":CHANnel{}:IMPedance FIFTy".format(self.channels[0]))
        self._write(":CHANnel{}:IMPedance FIFTy".format(self.channels[1]))
        # set in high resolution mode
        self._write(":ACQuire:TYPE HRESolution")
        # set individual channel settings
        self._write(":WAVeform:SOURce CHANnel{}".format(self.channels[0]))
        self._write(":WAVeform:FORMat WORD")
        self._write(":WAVeform:SOURce CHANnel{}".format(self.channels[1]))
        self._write(":WAVeform:FORMat WORD")
        # set full scale vertical range in volts for each channel
        self._set_range(0, 5)
        self._set_range(1, 5)

        if application == "pulse_data":
            assert len(self.channels) == 2, \
                "Only two channels allowed, one for I and another for Q"
            # set the time range (10xs the time per division)
            self._write(":TIMebase:RANGe 500e-6")

        elif application == "noise_data":
            # set the time range (10xs the time per division)
            self._write(":TIMebase:RANGe 500e-6")

        elif application == "sweep_data":
            # set the time range (10xs the time per division)
            self._write(":TIMebase:RANGe 1e-3")
            # remove trigger from the data range
            self._set_trigger(0, 5, 1)
        sleep(1)

    def take_pulse_data(self, n_triggers):
        # 500 * 500 us = 250 ms of data
        self._auto_range(search_length=500, max_range=True, max_iteration=1)
        self._autoscale_trigger(search_length=500, n_sigma=5)
        self._set_sweep_mode("normal")

        data_I = np.zeros((n_triggers, self.samples_per_channel))
        data_Q = np.zeros((n_triggers, self.samples_per_channel))
        for index in range(n_triggers):
            I_voltages, Q_voltages = self._get_data()
            data_I[index, :] = I_voltages
            data_Q[index, :] = Q_voltages

        return data_I, data_Q

    def take_noise_data(self, n_triggers):
        # find appropriate range and offset for the scope
        self._auto_range()
        # remove trigger from the data range
        self._set_trigger(0, 5, 1)

        data_I = np.zeros((n_triggers, self.samples_per_channel))
        data_Q = np.zeros((n_triggers, self.samples_per_channel))
        for index in range(n_triggers):
            rand_time = np.random.random_sample() * .001  # no longer than a millisecond
            sleep(rand_time)
            I_voltages, Q_voltages = self._get_data()
            data_I[index, :] = I_voltages
            data_Q[index, :] = Q_voltages

        return data_I, data_Q

    def take_iq_point(self):
        # find appropriate range and offset for the scope
        self._auto_range()
        # take data with those settings
        I_voltages, Q_voltages = self._get_data()
        # combine I and Q signals
        data = np.median(I_voltages) + 1j * np.median(Q_voltages)
        return data

    def close(self):
        self.session.close()

    def reset(self):
        self._write("*RST")
        sleep(1)

    def _write(self, *args, **kwargs):
        self.session.write(*args, **kwargs)

    def _read(self, *args, **kwargs):
        return self.session.read(*args, **kwargs)

    def _query(self, *args, **kwargs):
        return self.session.query(*args, **kwargs)

    def _query_ascii_values(self, *args, **kwargs):
        return self.session.query_ascii_values(*args, **kwargs)

    def _query_binary_values(self, *args, **kwargs):
        return self.session.query_binary_values(*args, **kwargs)

    def _get_data(self):
        self._collect_trigger()
        self._select_channel(0)
        I_voltages = self._get_voltages()
        self._select_channel(1)
        Q_voltages = self._get_voltages()

        return I_voltages, Q_voltages

    def _auto_range(self, search_length=1, max_range=False, max_iteration=20):
        scale = 20
        stop_condition = False
        itteration = 0
        dI_old = self._get_range(0)
        dQ_old = self._get_range(1)
        Ic_old = self._get_offset(0)
        Qc_old = self._get_offset(1)

        while not stop_condition:
            I_median, I_std, Q_median, Q_std = [], [], [], []
            for _ in range(search_length):
                self._single_trigger()
                I_voltages, Q_voltages = self._get_data()
                I_median.append(np.median(I_voltages))
                Q_median.append(np.median(Q_voltages))
                if max_range:
                    I_std.append((np.max(I_voltages) - np.min(I_voltages)) / scale)
                    Q_std.append((np.max(Q_voltages) - np.min(Q_voltages)) / scale)
                else:
                    I_std.append(np.std(I_voltages))
                    Q_std.append(np.std(Q_voltages))
            # get new voltage range
            if np.mean(I_std) == 0:
                dI = dI_old * 2
            else:
                if max_range:
                    dI = max(3 * scale * np.max(I_std), 0.2)
                else:
                    dI = max(scale * np.mean(I_std), 0.2)
            if np.mean(Q_std) == 0:
                dQ = dQ_old * 2 
            else:
                if max_range:
                    dQ = max(3 * scale * np.max(Q_std), 0.2)
                else:
                    dQ = max(scale * np.mean(Q_std), 0.2)

            # get new voltage offset
            Ic = np.mean(I_median)
            Qc = np.mean(Q_median)
            
            self._set_range(0, dI)
            self._set_range(1, dQ)
            self._set_offset(0, Ic)
            self._set_offset(1, Qc)
            
            # See what we set it to (scope rounds to the nearest legal value)
            dI = self._get_range(0)
            dQ = self._get_range(1)
            Ic = self._get_offset(0)
            Qc = self._get_offset(1)

            # calculate offset and range changes
            range_change = (np.abs(dI_old - dI) / (np.abs(dI) + 0.001) +
                            np.abs(dQ_old - dQ) / (np.abs(dQ) + 0.001))
            offset_change = (np.abs(Ic_old - Ic) / (np.abs(Ic) + 0.001) +
                             np.abs(Qc_old - Qc) / (np.abs(Qc) + 0.001))

            # save offsets and ranges
            dI_old, dQ_old, Ic_old, Qc_old = dI, dQ, Ic, Qc
            
            std_not_zero = np.logical_and(np.mean(I_std) != 0, np.mean(Q_std) != 0)
            small_change = np.logical_and(offset_change < 0.15, range_change < 0.15)

            itteration += 1
            stop_condition = np.logical_or(itteration >= max_iteration,
                                           np.logical_and(std_not_zero, small_change))

    def _get_voltages(self):
        # get preamble
        preamble = self._query_ascii_values(":WAVeform:PREamble?")
        # wav_form = preamble[0]
        # acq_type = preamble[1]
        # wfmpts = preamble[2]
        # avgcnt = preamble[3]
        # x_increment = preamble[4]
        # x_origin = preamble[5]
        # x_reference = preamble[6]
        y_increment = preamble[7]
        y_origin = preamble[8]
        y_reference = preamble[9]
        # record first channel
        values = self._query_binary_values(":WAVeform:DATA?", datatype="H",
                                           container=np.array,
                                           is_big_endian=True)
        # convert to voltages
        voltages = ((values - y_reference) * y_increment) + y_origin
        return voltages

    def _autoscale_trigger(self, n_sigma=4, search_length=50):
        # compute the median absolute deviation for a bunch of acquisitions
        I_max, I_min, I_median, I_mad = [], [], [], []
        Q_max, Q_min, Q_median, Q_mad = [], [], [], []
        for _ in range(search_length):
            self._single_trigger()
            I_voltages, Q_voltages = self._get_data()
            I_mad.append(np.median(np.abs(I_voltages - np.median(I_voltages))))
            I_median.append(np.median(I_voltages))
            I_max.append(np.max(I_voltages))
            I_min.append(np.min(I_voltages))
            Q_mad.append(np.median(np.abs(Q_voltages - np.median(Q_voltages))))
            Q_median.append(np.median(Q_voltages))
            Q_max.append(np.max(Q_voltages))
            Q_min.append(np.min(Q_voltages))

        # the index with the smallest median absolute deviation has the least
        # pulse contamination
        ind = np.argmin(I_mad)
        I_median = I_median[ind]
        I_mad = I_mad[ind]
        ind = np.argmin(Q_mad)
        Q_median = Q_median[ind]
        Q_mad = Q_mad[ind]
        # determine max and min deviations from the median for each channel
        I_pos = max(I_max) - I_median
        I_neg = I_median - min(I_min)
        Q_pos = max(Q_max) - Q_median
        Q_neg = Q_median - min(Q_min)
        # choose the channel that has the biggest pulses
        ind = np.argmax([I_pos, I_neg, Q_pos, Q_neg])
        sigma = 1.4826 * np.array([I_mad, I_mad, Q_mad, Q_mad])
        median = np.array([I_median, I_median, Q_median, Q_median])
        slope = np.array([1, -1, 1, -1])
        level = slope * n_sigma * sigma + median
        channel = [0, 0, 1, 1]

        # set the trigger channel, level and slope
        self._set_trigger(channel[ind], level[ind], slope[ind])

    def _set_trigger(self, channel, level, slope):
        if slope > 0:
            slope = "POS"
        else:
            slope = "NEG"
        self._write(":TRIGger:SOURce CHANnel{}".format(self.channels[channel]))
        self._write(":TRIGger:LEVel {:.6f}".format(level))
        self._write(":TRIGger:SLOPe {}".format(slope))

    def _collect_trigger(self):
        self._write(":DIGitize CHANnel{}, CHANnel{}".format(*self.channels))

    def _select_channel(self, channel):
        self._write(":WAVeform:SOURce CHANnel{}".format(self.channels[channel]))

    def _set_range(self, channel, new_range):
        self._write(":CHANnel{}:RANGe {:.3f} V"
                    .format(self.channels[channel], new_range))

    def _set_offset(self, channel, new_offset):
        self._write(":CHANnel{}:OFFSet {:.3f} V"
                    .format(self.channels[channel], new_offset))

    def _single_trigger(self):
        self._write(":SINGle")

    def _get_range(self, channel):
        return float(self._query(":CHANnel{}:RANGe?".format(self.channels[channel])))
    
    def _get_offset(self, channel):
        return float(self._query(":CHANnel{}:OFFSet?".format(self.channels[channel])))
    
    def _set_sweep_mode(self, mode):
        if mode == "auto":
            self._write(":TRIGger:SWEep AUTO")
        elif mode == "normal":
            self._write(":TRIGger:SWEep NORMal")
        else:
            message = "{} is not a valid sweep mode. Use 'auto' or 'normal'"
            raise ValueError(message.format(mode))