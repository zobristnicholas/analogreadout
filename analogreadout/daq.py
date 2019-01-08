import warnings
import importlib
from datetime import datetime
from analogreadout.configurations import config
from analogreadout.instruments.sensors import NotASensor
from analogreadout.custom_warnings import ConnectionWarning
from analogreadout.instruments.attenuators import NotAnAttenuator
from analogreadout.functions import take_noise_data, do_iq_sweep, take_pulse_data

DEFAULT_CONFIG = "JPL"


def get_procedure(procedure):
    library = importlib.import_module("analogreadout.procedures")
    return getattr(library, procedure)
    
    
def get_instrument(dictionary):
    location = dictionary['location']
    instrument = dictionary['instrument']
    library = importlib.import_module("analogreadout.instruments." + location)
    return getattr(library, instrument)(*dictionary['arguments'])
    

class DAQ:
    """
    Data Acquisition System:
    Class for holding all of the instrument objects defined in the configuration
    dictionary. Also holds methods to initialize, close and reset all instruments at once.
    Data taking methods are take_noise_data(), do_iq_sweep(), and take_pulse_data().    
    """
    def __init__(self, configuration=None):
        # load configuration
        if configuration is None:
            self.config = config(DEFAULT_CONFIG)
        else:
            self.config = config(configuration)
        
        # initialize all instruments as None
        self.dac = None  # digital to analog converter
        self.adc = None  # analog to digital converter
        self.dac_atten = None  # dac attenuator
        self.adc_atten = None  # adc attenuator
        self.thermometer = None  # device thermometer
        self.primary_amplifier = None  # HEMT amplifier or para-amp 

        # set the instruments specified in the configuration
        for key, value in self.config['dac'].items():
            if key == "dac":
                self.dac = get_instrument(value)
            elif key == "attenuator":
                self.dac_atten = get_instrument(value)

        for key, value in self.config['adc'].items():
            if key == "adc":
                self.adc = get_instrument(value)
            elif key == "attenuator":
                self.adc_atten = get_instrument(value)

        for key, value in self.config['sensors'].items():
            if key == "thermometer":
                self.thermometer = get_instrument(value)
            elif key == "primary_amplifier":
                self.primary_amplifier = get_instrument(value)
                
        # if the instrument wasn't initialized set it to a dummy NotAnInstrument class
        if self.adc is None or self.dac is None:
            raise ValueError("configuration must specify an adc and dac")
        if self.dac_atten is None:
            self.dac_atten = NotAnAttenuator("DAC")
        if self.adc_atten is None:
            self.adc_atten = NotAnAttenuator("ADC")
        if self.thermometer is None:
            self.thermometer = NotASensor("Thermometer")
        if self.primary_amplifier is None:
            self.primary_amplifier = NotASensor("Primary Amplifier")
            
        # only display missing instrument warnings once
        # warnings.simplefilter("once", ConnectionWarning)
            
        
    def procedure(self, procedure_type):
        """
        Return the procedure class of a given type.
        Args:
            procedure_type: sweep, noise, or pulse (str)
        """
        library = importlib.import_module("analogreadout.procedures")
        procedure_class = getattr(library, self.config["procedures"][procedure_type])
        procedure_class.connect_daq(self.daq)
        return procedure_class
        

    def take_noise_data(self, *args, **kwargs):
        """
        Take noise data
        Args:
            frequency: frequency [GHz]
            dac_atten: dac attenuation [dB]
            n_triggers: number of noise triggers (int)
            directory: folder where data should be saved (string)
            power: dac power [dB] (optional, should be set by configuration)
            adc_atten: adc attenuation [dB] (optional, defaults to 0)
            sample_rate: sample rate of adc (float, defaults to 2e6 Hz)
            verbose: print information about the system (bool, defaults to True)
        
        Returns:
            file_path: full path where the data was saved
        """
        if "power" not in kwargs.keys():
            kwargs.update({"power": self.config['dac']['dac']['power']})
        return take_noise_data(self.daq, *args, **kwargs)

    def do_iq_sweep(self, *args, **kwargs):
        """
        Take an iq sweep
        Args:
            center: center frequency [GHz]
            span: sweep span [GHz]
            dac_atten: dac attenuation [dB]
            n_points: number of points in the sweep (int)
            directory: folder where data should be saved (string)
            power: dac power [dB] (optional, should be set by configuration)
            adc_atten: adc attenuation [dB] (optional, defaults to 0)
            verbose: print information about the system (bool, defaults to True)
            
        Returns:
            file_path: full path where the data was saved.
        """
        if "power" not in kwargs.keys():
            kwargs.update({"power": self.config['dac']['dac']['power']})
        return do_iq_sweep(self.daq, *args, **kwargs)

    def take_pulse_data(self, *args, **kwargs):
        """
        Take pulse data
        Args:
            frequency: frequency [GHz]
            dac_atten: dac attenuation [dB]
            n_triggers: number of noise triggers (int)
            directory: folder where data should be saved (string)
            power: dac power [dB] (optional, should be set by configuration)
            adc_atten: adc attenuation [dB] (optional, defaults to 0)
            sample_rate: sample rate of adc (float, defaults to 2e6 Hz)
            verbose: print information about the system (bool, defaults to True)
            
        Returns:
            file_path: full path where the data was saved
        """
        if "power" not in kwargs.keys():
            kwargs.update({"power": self.config['dac']['dac']['power']})
        return take_pulse_data(self.daq, *args, **kwargs)
        
    def initialize(self, application, frequency, power=None, dac_atten=0, adc_atten=0):
        """
        Initialize all of the instruments according to their initialize methods
        Args:
            application: type of acquisition to send to the ADC defining the data taking
                         application (string)
            frequency: frequency to output from the DAC [GHz]
            power: power to output from the DAC [dBm] (optional, defaluts to config value)
            dac_atten: DAC attenuation [dB] (optional, defaults to 0)
            adc_atten: ADC attenuation [dB] (optional, defaults to 0)
        """
        self.dac_atten.initialize(dac_atten)
        self.adc_atten.initialize(adc_atten)
        self.dac.initialize(frequency, power)
        self.adc.initialize(application)
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
        
    def sensor_states(self, metadata):
        """
        Returns a dictionary of sensor data with a timestamp
        """
        state ={"timestamp": datetime.now().strftime('%Y%m%d_%H%M%S'),
                "thermometer": self.thermometer.read_value(),
                "primary_amplifier": self.primary_amplifier.read_value()}
        return state