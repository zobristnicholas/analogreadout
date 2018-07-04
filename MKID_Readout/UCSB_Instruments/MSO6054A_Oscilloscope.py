import visa


class MSO6054A_Oscilloscope:
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

    def initialize(self, application, channels=[1, 3]):
        # recast channels as integers
        channels = [int(channel) for channel in channels]
        self.channels = channels
        self.write("*RST")
        self.write("*CLS")
        if application == "pulse_data":
            assert len(channels) == 2, \
                "Only two channels allowed, one for I and another for Q"

            # set full scale vertical range in volts for each channel
            self.write(":CHANnel{}:RANGe 0.016\n".format(channels[0]))
            self.write(":CHANnel{}:RANGe 0.016\n".format(channels[1]))

            # set each channel at 50 Ohm impedance
            self.write(":CHANnel{}:IMPedance FIFTy".format(channels[0]))
            self.write(":CHANnel{}:IMPedance FIFTy".format(channels[1]))

            # set the time range (10xs the time per division)
            self.write(":TIMebase:RANGe 500e-6")
            # set the display reference to the center
            # self.write(":TIMebase:REFerence CENTer")
            # # set the trigger mode
            # self.write(":TRIGger:MODE EDGE")
            # # set the trigger source channel
            # self.write(":TRIGger:EDGE:SOURce CHANnel{}"
            #            .format(str(int(channels[0]))))
            # # set the trigger coupling
            # self.write(":TRIGger:EDGE:COUPling DC\n")
            # # set the trigger level
            # self.write(":TRIGger:EDGE:LEVel 2")
            # # set the trigger slope polarity
            # self.write(":TRIGger:EDGE:SLOPe POSitive")

            # set in high resolution mode
            self.write(":ACQuire:TYPE HRESolution")

            # set individual channel settings
            self.write(":WAVeform:SOURce CHANnel{}".format(channels[0]))
            self.write(":WAVeform:FORMat WORD")
            self.write(":WAVeform:SOURce CHANnel{}".format(channels[1]))
            self.write(":WAVeform:FORMat WORD")
            self.write(":WAVeform:POINts:MODE RAW")

    def take_pulse_data(n_traces):
        # take data
        I_traces = np.zeros((n_traces, 1000))
        Q_traces = np.zeros((n_traces, 1000))
        for index in range(n_traces):
            # collect trigger
            self.write(":DIGitize CHANnel{}, CHANnel{}".format(*self.channels))
            # record first channel
            self.write(":WAVeform:SOURce CHANnel{}".format(self.channels[0]))
            I_traces[index, :] = self.query_binary_values(":WAVeform:DATA?", datatype="H",
                                                          container=np.array,
                                                          is_big_endian=True)
            # record second channel
            self.write(":WAVeform:SOURce CHANnel {}".format(self.channels[1]))
            Q_traces[index, :] = self.query_binary_values(":WAVeform:DATA?", datatype="H",
                                                          container=np.array,
                                                          is_big_endian=True)

        # get preamble
        preamble = self.query_ascii_values(":WAVeform:PREamble?")
        wav_form = preamble[0]
        acq_type = preamble[1]
        wfmpts = preamble[2]
        avgcnt = preamble[3]
        x_increment = preamble[4]
        x_origin = preamble[5]
        x_reference = preamble[6]
        y_increment = preamble[7]
        y_origin = preamble[8]
        y_reference = preamble[9]

        # convert to voltages
        I_voltage = ((I_traces - y_reference) * y_increment) + y_origin
        Q_voltage = ((Q_traces - y_reference) * y_increment) + y_origin

        # combine I and Q signals
        data = I_voltage + 1j * Q_voltage

        return data

    def take_iq_point():
        data = self.query(":WAVEFORM:DATA?")

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
