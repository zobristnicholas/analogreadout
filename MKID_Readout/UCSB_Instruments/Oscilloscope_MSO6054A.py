import visa
import numpy as np
from time import sleep


class Oscilloscope_MSO6054A:
    def __init__(self, address):
        try:
            resourceManager = visa.ResourceManager()
        except:
            resourceManager = visa.ResourceManager('@py')
        self.session = resourceManager.open_resource(address)
        identity = self.query_ascii_values("*IDN?", 's')
        print("Connected to:", identity[0])
        print("Model Number:", identity[1])
        print("Serial Number:", identity[2])
        print("System Version:", identity[3])

    def initialize(self, application, channels=(1, 3)):
        # recast channels as integers
        channels = [int(channel) for channel in channels]
        self.channels = channels
        self.reset()
        self.write("*CLS")
        if application == "pulse_data":
            assert len(channels) == 2, \
                "Only two channels allowed, one for I and another for Q"

            # set full scale vertical range in volts for each channel
            self.write(":CHANnel{}:RANGe 0.016 V".format(channels[0]))
            self.write(":CHANnel{}:RANGe 0.016 V".format(channels[1]))

            # set each channel at 50 Ohm impedance
            self.write(":CHANnel{}:IMPedance FIFTy".format(channels[0]))
            self.write(":CHANnel{}:IMPedance FIFTy".format(channels[1]))

            # set the time range (10xs the time per division)
            self.write(":TIMebase:RANGe 500e-6")


            # set in high resolution mode
            self.write(":ACQuire:TYPE HRESolution")

            # set individual channel settings
            self.write(":WAVeform:SOURce CHANnel{}".format(channels[0]))
            self.write(":WAVeform:FORMat WORD")
            self.write(":WAVeform:SOURce CHANnel{}".format(channels[1]))
            self.write(":WAVeform:FORMat WORD")
            
            # set trigger to normal
            self.write(":TRIGger:SWEep NORMal")
            
            
        elif application == "noise_data":

            # set each channel at 50 Ohm impedance
            self.write(":CHANnel{}:IMPedance FIFTy".format(channels[0]))
            self.write(":CHANnel{}:IMPedance FIFTy".format(channels[1]))
            
            # set the time range (10xs the time per division)
            self.write(":TIMebase:RANGe 500e-6")

            # set in high resolution mode
            self.write(":ACQuire:TYPE HRESolution")

            # set individual channel settings
            self.write(":WAVeform:SOURce CHANnel{}".format(channels[0]))
            self.write(":WAVeform:FORMat WORD")
            self.write(":WAVeform:SOURce CHANnel{}".format(channels[1]))
            self.write(":WAVeform:FORMat WORD")
       
        elif application == "iq_sweep":
            # set full scale vertical range in volts for each channel
            self.write(":CHANnel{}:RANGe 3 V".format(channels[0]))
            self.write(":CHANnel{}:RANGe 3 V".format(channels[1]))
            # set each channel at 50 Ohm impedance
            self.write(":CHANnel{}:IMPedance FIFTy".format(channels[0]))
            self.write(":CHANnel{}:IMPedance FIFTy".format(channels[1]))
            # set the time range (10xs the time per division)
            self.write(":TIMebase:RANGe 1e-3")
            # set individual channel settings
            self.write(":WAVeform:SOURce CHANnel{}".format(channels[0]))
            self.write(":WAVeform:FORMat WORD")
            self.write(":WAVeform:SOURce CHANnel{}".format(channels[1]))
            self.write(":WAVeform:FORMat WORD")
        sleep(1)

    def take_pulse_data(self, offset, volts_per_div, n_triggers, trig_level, slope,
                        trig_chan):
        
        d1 = volts_per_div[0] * 8
        d2 = volts_per_div[1] * 8
        
        self.write(":CHANnel{}:RANGe {:.6f} V".format(self.channels[0], d1))
        self.write(":CHANnel{}:RANGe {:.6f} V".format(self.channels[1], d2))
        self.write(":CHANnel{}:OFFSet {:.6f} V".format(self.channels[0], offset[0]))
        self.write(":CHANnel{}:OFFSet {:.6f} V".format(self.channels[1], offset[1]))
        self.write(":TRIGger:SOURce CHANnel{}".format(trig_chan))
        self.write(":TRIGger:LEVel {:.6f}".format(trig_level))
        self.write(":TRIGger:SLOPe {}".format(slope))
        
        data_I = np.zeros((n_triggers, 1000))
        data_Q = np.zeros((n_triggers, 1000))
        for index in range(n_triggers):
            rand_time = np.random.random_sample() * .001 # no longer than a milisecond
            sleep(rand_time)
            I_voltages, Q_voltages = self.take_data()
            data_I[index, :] = I_voltages
            data_Q[index, :] = Q_voltages
            
        return data_I, data_Q

    def take_noise_data(self, offset, volts_per_div, n_triggers):
        
        d1 = volts_per_div[0] * 8
        d2 = volts_per_div[1] * 8
        
        self.write(":CHANnel{}:RANGe {:.6f} V".format(self.channels[0], d1))
        self.write(":CHANnel{}:RANGe {:.6f} V".format(self.channels[1], d2))
        self.write(":CHANnel{}:OFFSet {:.6f} V".format(self.channels[0], offset[0]))
        self.write(":CHANnel{}:OFFSet {:.6f} V".format(self.channels[1], offset[1]))
        
        data_I = np.zeros((n_triggers, 1000))
        data_Q = np.zeros((n_triggers, 1000))
        for index in range(n_triggers):
            rand_time = np.random.random_sample() * .001 # no longer than a milisecond
            sleep(rand_time)
            I_voltages, Q_voltages = self.take_data()
            data_I[index, :] = I_voltages
            data_Q[index, :] = Q_voltages
            
        return data_I, data_Q
        
        
    def take_iq_point(self):
        # collect trigger
        self.write(":DIGitize CHANnel{}, CHANnel{}".format(*self.channels))
        # switch to fist channel
        self.write(":WAVeform:SOURce CHANnel{}".format(self.channels[0]))
        # get preamble
        preamble = self.query_ascii_values(":WAVeform:PREamble?")
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
        I_values = self.query_binary_values(":WAVeform:DATA?", datatype="H",
                                            container=np.array,
                                            is_big_endian=True)
        # convert to voltages
        I_voltages = ((I_values - y_reference) * y_increment) + y_origin
        
        # switch to second channel
        self.write(":WAVeform:SOURce CHANnel{}".format(self.channels[1]))
        # get preamble
        preamble = self.query_ascii_values(":WAVeform:PREamble?")
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
        # record second channel
        Q_values = self.query_binary_values(":WAVeform:DATA?", datatype="H",
                                            container=np.array,
                                            is_big_endian=True)
        # convert to voltages
        Q_voltages = ((Q_values - y_reference) * y_increment) + y_origin
        
        # get new voltage range
        dI = max(1.2 * (max(I_voltages) - min(I_voltages)), 0.05)
        dQ = max(1.2 * (max(Q_voltages) - min(Q_voltages)), 0.05)
        
        # get new voltage offset
        Ic = np.mean(I_voltages)
        Qc = np.mean(Q_voltages)
        
        # reset the range and offset
        self.write(":CHANnel{}:RANGe {:.3f} V".format(self.channels[0], dI))
        self.write(":CHANnel{}:RANGe {:.3f} V".format(self.channels[1], dQ))
        self.write(":CHANnel{}:OFFSet {:.3f} V".format(self.channels[0], Ic))
        self.write(":CHANnel{}:OFFSet {:.3f} V".format(self.channels[1], Qc))

        # collect trigger
        self.write(":DIGitize CHANnel{}, CHANnel{}".format(*self.channels))
        # switch to fist channel
        self.write(":WAVeform:SOURce CHANnel{}".format(self.channels[0]))
        # get preamble
        preamble = self.query_ascii_values(":WAVeform:PREamble?")
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
        I_values = self.query_binary_values(":WAVeform:DATA?", datatype="H",
                                            container=np.array,
                                            is_big_endian=True)
        # convert to voltages
        I_voltages = ((I_values - y_reference) * y_increment) + y_origin
        
        # switch to second channel
        self.write(":WAVeform:SOURce CHANnel{}".format(self.channels[1]))
        # get preamble
        preamble = self.query_ascii_values(":WAVeform:PREamble?")
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
        # record second channel
        Q_values = self.query_binary_values(":WAVeform:DATA?", datatype="H",
                                            container=np.array,
                                            is_big_endian=True)
        # convert to voltages
        Q_voltages = ((Q_values - y_reference) * y_increment) + y_origin
        
        # combine I and Q signals
        data = np.median(I_voltages) + 1j * np.median(Q_voltages)
        
        return data

    def write(self, *args, **kwargs):
        self.session.write(*args, **kwargs)

    def read(self, *args, **kwargs):
        return self.session.read(*args, **kwargs)

    def query(self, *args, **kwargs):
        return self.session.query(*args, **kwargs)

    def query_ascii_values(self, *args, **kwargs):
        return self.session.query_ascii_values(*args, **kwargs)

    def query_binary_values(self, *args, **kwargs):
        return self.session.query_binary_values(*args, **kwargs)
    
    def take_data(self):
        # collect trigger
        self.write(":DIGitize CHANnel{}, CHANnel{}".format(*self.channels))
        # switch to fist channel
        self.write(":WAVeform:SOURce CHANnel{}".format(self.channels[0]))
        # get preamble
        preamble = self.query_ascii_values(":WAVeform:PREamble?")
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
        I_values = self.query_binary_values(":WAVeform:DATA?", datatype="H",
                                            container=np.array,
                                            is_big_endian=True)
        # convert to voltages
        I_voltages = ((I_values - y_reference) * y_increment) + y_origin
        
        # switch to second channel
        self.write(":WAVeform:SOURce CHANnel{}".format(self.channels[1]))
        # get preamble
        preamble = self.query_ascii_values(":WAVeform:PREamble?")
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
        # record second channel
        Q_values = self.query_binary_values(":WAVeform:DATA?", datatype="H",
                                            container=np.array,
                                            is_big_endian=True)
        # convert to voltages
        Q_voltages = ((Q_values - y_reference) * y_increment) + y_origin
        
        return I_voltages, Q_voltages
    def close(self):
        self.session.close()

    def reset(self):
        self.write("*RST")
        sleep(1)