import os
import visa
import logging
import importlib
import numpy as np
from time import sleep
from datetime import datetime
from pymeasure.experiment import Parameter
from analogreadout.configurations import config
from analogreadout.instruments.sensors import NotASensor
from analogreadout.instruments.attenuators import NotAnAttenuator
from analogreadout.instruments.resistance_bridges import NotAThermometer
from analogreadout.functions import take_noise_data, do_iq_sweep, take_pulse_data

log = logging.getLogger(__name__)
log.addHandler(logging.NullHandler())

DEFAULT_CONFIG = "JPL"


def get_procedure(procedure):
    library = importlib.import_module("analogreadout.procedures")
    return getattr(library, procedure)
    
    
def get_instrument(dictionary):
    location = dictionary['location']
    instrument = dictionary['instrument']
    library = importlib.import_module("analogreadout.instruments." + location)
    try:
        return getattr(library, instrument)(*dictionary['arguments'])
    except visa.VisaIOError as error:
        busy = (error.error_code == visa.constants.VI_ERROR_MACHINE_NAVAIL or
                error.error_code == visa.constants.StatusCode.error_resource_busy)
        if busy:
            message = ("The '{}' instrument is busy. If another program is using it, try "
                       "closing it or moving the instrument to a non-serial connection")
            log.error(message.format(instrument))
        else:
            message = "Error loading the '{}' instrument: " + error
            log.error(message.format(instrument))
    except:
        message = "Error loading the '{}' instrument: " + error
        log.error(message.format(instrument))
    

class DAQ:
    """
    Data Acquisition System:
    Class for holding all of the instrument objects defined in the configuration
    dictionary. Also holds methods to initialize, close and reset all instruments at once.
    Data taking methods are selected from the run() method.
    """
    def __init__(self, configuration=None):
        # load configuration
        if configuration is None:
            self.config = config(DEFAULT_CONFIG)
        else:
            self.config = config(configuration)
        # initialize all instruments as None
        self.instrument_names = ('dac_atten',  # digital to analog converter
                                 'adc_atten',  # analog to digital converter
                                 'dac',  # dac attenuator
                                 'adc',  # adc attenuator
                                 'thermometer',  # device thermometer
                                 'primary_amplifier')  # first amplifier in the chain
        for name in self.instrument_names:
            setattr(self, name, None)
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
            self.thermometer = NotAThermometer()
        if self.primary_amplifier is None:
            self.primary_amplifier = NotASensor("Primary Amplifier")
        # set a flag that tracks if the all the instruments have been closed
        self.closed = False
            
        
    def procedure_class(self, procedure_type):
        """
        Return the procedure class of a given type.
        Args:
            procedure_type: sweep, noise, or pulse (str)
        """
        library = importlib.import_module("analogreadout.procedures")
        ProcedureClass = getattr(library, self.config["procedures"][procedure_type])
        ProcedureClass.connect_daq(self)
        return ProcedureClass
    
    def run(self, procedure_type, file_name_kwargs={}, stop=None, **kwargs):
        """
        Take data for the given procedure_type. The procedure class is defined in the
        configuration file.
        Args:
        procedure_type: sweep, noise or pulse (str)
        file_name_kwargs: kwargs to pass to procedure.file_name() after instantiation
        stop: method to monkey patch into procedure.stop  (used in chained procedures)
        **kwargs: procedure parameters (set to the defaults if not specified)
        """
        # get proceedure class
        Procedure = self.procedure_class(procedure_type)
        # overload the default parameter value and set it's value
        for key, value in kwargs.items():
            getattr(Procedure, key).default = value
            getattr(Procedure, key).value = value
        # check that all parameters have a default
        for name in dir(Procedure):
            parameter = getattr(Procedure, name)
            if isinstance(parameter, Parameter):
                message = "{} is not an optional parameter. No default is specified"
                assert parameter.default is not None, message.format(name)
        # run procedure
        procedure = Procedure()
        if stop is not None:
            procedure.stop = stop
        if file_name_kwargs.get("prefix", None) is None:
            file_name_kwargs["prefix"] = procedure_type
        procedure.file_name(**file_name_kwargs)
        try:
            procedure.startup()
            procedure.execute()
        finally:
            procedure.shutdown()
        # return the saved file name
        try:
            file_name = os.path.join(procedure.directory, procedure.file_name())
        except:
            file_name = None
        return file_name   
        

    def take_noise_data(self, *args, **kwargs):
        """
        Take noise data.
        Args:
            frequency: frequency [GHz]
            dac_atten: dac attenuation [dB]
            n_triggers: number of noise triggers (int)
            directory: folder where data should be saved (string)
            power: dac power [dB] (optional, should be set by configuration)
            adc_atten: adc attenuation [dB] (optional, defaults to 0)
            sample_rate: sample rate of adc [Hz] (optional, defaults to 2e6)
            verbose: print information about the system (optional, defaults to True)
        
        Returns:
            file_path: full path where the data was saved
        """
        if "power" not in kwargs.keys():
            kwargs.update({"power": self.config['dac']['dac']['power']})
        return take_noise_data(self.daq, *args, **kwargs)


    def take_pulse_data(self, *args, **kwargs):
        """
        Take pulse data.
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
        
    def initialize(self, frequency, power=None, dac_atten=0, adc_atten=0,
                   sample_rate=None, n_samples=None, channels=None):
        """
        Initialize all of the instruments according to their initialize methods.
        Args:
            application: type of acquisition to send to the ADC defining the data taking
                         application (string)
            frequency: frequency to output from the DAC [GHz]
            power: power to output from the DAC [dBm] (optional, defaluts to config value)
            dac_atten: DAC attenuation [dB] (optional, defaults to 0)
            adc_atten: ADC attenuation [dB] (optional, defaults to 0)
            sample_rate: ADC sample rate [Hz] (optional, default depends on ADC)
                The sample rate may be variable or only able to take one value depending
                on the hardware.
            n_samples: samples per ADC acquisition (optional, default depends on ADC)
            channels: ADC channels to take data with (optional, default depends on ADC)
        """
        self.dac_atten.initialize(dac_atten)
        self.adc_atten.initialize(adc_atten)
        if power is None:
            power = self.config['dac']['dac']['power']
        self.dac.initialize(frequency, power)
        self.adc.initialize(sample_rate=sample_rate, n_samples=n_samples,
                            channels=channels)
        self.thermometer.initialize()
        self.primary_amplifier.initialize()
        sleep(1)

    def close(self):
        """
        Close all of the instruments according to their close methods.
        """
        if self.closed:
            log.debug("The DAQ is already closed")
            return
        problem = False
        for name in self.instrument_names:
            instrument = getattr(self, name)
            try:
                instrument.close()
            except (AttributeError, visa.InvalidSession):
                pass  # ignore if close() doesn't exist or the resource is already closed
            except Exception as error:
                problem = True
                message = "The '{}' instrument was unable to close: "
                log.warning(message.format(instrument.__class__.__name__) + error)
        if problem:
            log.warning("There was a problem shutting down the DAQ")
        else:
            self.closed = True
            log.info("The DAQ was properly shut down")
        

    def reset(self):
        """
        Reset all of the instruments according to their reset methods.
        """
        for name in self.instrument_names:
            instrument = getattr(self, name)
            try:
                instrument.reset()
            except AttributeError:
                pass
            except Exception as error:
                message = "The '{}' instrument was unable to reset: "
                log.warning(message.format(instrument.__class__.__name__) + error)
        
    def system_state(self):
        """
        Returns a dictionary defining the system state with a timestamp
        """
        temperatures = []
        resistances = []
        for _ in range(10):
            temperatures.append(self.thermometer.temperature)
            resistances.append(self.thermometer.resistance)
            sleep(self.thermometer.WAIT_MEASURE)
        
        thermometer = {'channel': self.thermometer.channel,
                       'temperature': (np.mean(temperatures), np.std(temperatures)),
                       'resistance': (np.mean(resistances), np.std(resistances))}
            
        state ={"timestamp": datetime.now().strftime('%Y%m%d_%H%M%S'),
                "thermometer": thermometer,
                "primary_amplifier": self.primary_amplifier.read_value()}
        return state