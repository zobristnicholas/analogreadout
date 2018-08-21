import importlib
from MKID_Readout.instruments.attenuators import NotAnAttenuator
from MKID_Readout.instruments.sensors import NotASensor


def get_instrument(dictionary):
    location = dictionary['location']
    instrument = dictionary['instrument']
    library = importlib.import_module("MKID_Readout.instruments.hardware." +
                                      location)
    return getattr(library, instrument)(*dictionary['arguments'])


class DAQ:
    """
    Class for holding all of the instrument objects defined in the configuration
    dictionary. Also holds methods to initialize, close and reset all instruments at once.
    """
    def __init__(self, configuration):
        # initialize all possible instrument types as NotAnInstrument class
        # so that code runs if one or more pieces do not exist
        self.dac = None
        self.adc = None
        self.dac_atten = NotAnAttenuator("DAC")
        self.adc_atten = NotAnAttenuator("ADC")
        self.thermometer = NotASensor("Thermometer")
        self.primary_amplifier = NotASensor("Primary Amplifier")

        self.config = configuration

        for key, value in configuration['dac'].items():
            if key == "dac":
                self.dac = get_instrument(value)
            elif key == "attenuation":
                self.dac_atten = get_instrument(value)

        for key, value in configuration['adc'].items():
            if key == "adc":
                self.adc = get_instrument(value)
            elif key == "attenuation":
                self.adc_atten = get_instrument(value)

        for key, value in configuration['sensors'].items():
            if key == "thermometer":
                self.thermometer = get_instrument(value)
            elif key == "primary_amplifier":
                self.primary_amplifier = get_instrument(value)

        if self.adc is None or self.dac is None:
            raise ValueError("configuration must specify an adc and dac")

    def initialize(self, application, frequency, power, dac_atten=0, adc_atten=0):
        """
        Initialize all of the instruments according to their initialize methods
        Args:
            application: type of acquisition to send to the ADC defining the data taking
                         application (string)
            frequency: frequency to output from the DAC [GHz]
            power: power to output from the DAC [dBm]
            dac_atten: DAC attenuation [dB] (optional, defaults to 0)
            adc_atten: ADC attenuation [dB] (optional, defaults to 0)
        """
        self.dac_atten.initialize(dac_atten)
        self.adc_atten.initialize(adc_atten)
        self.dac.initialize(application)
        self.adc.initialize(frequency, power)
        self.thermometer.initialize()
        self.primary_amplifier.initialize()

    def close(self):
        """
        Close all of the instruments according to their close methods
        """
        self.dac_atten.close()
        self.adc_atten.close()
        self.dac.close()
        self.adc.close()
        self.thermometer.close()
        self.primary_amplifier.close()

    def reset(self):
        """
        Reset all of the instruments according to their reset methods
        """
        self.dac_atten.reset()
        self.adc_atten.reset()
        self.dac.reset()
        self.adc.reset()
        self.thermometer.reset()
        self.primary_amplifier.reset()
