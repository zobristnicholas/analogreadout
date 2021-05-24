import os
import yaml
import logging
import importlib
import numpy as np
import pyvisa as visa
import yaml
from pyvisa import constants
from time import sleep
from datetime import datetime
from pymeasure.experiment import Parameter
from analogreadout.instruments.sources import NotASource
from analogreadout.instruments.sensors import NotASensor
from analogreadout.instruments.attenuators import NotAnAttenuator
from analogreadout.instruments.resistance_bridges import NotAThermometer

log = logging.getLogger(__name__)
log.addHandler(logging.NullHandler())

DEFAULT_CONFIG = "ucsb"


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
        busy = (error.error_code == constants.VI_ERROR_MACHINE_NAVAIL or
                error.error_code == constants.StatusCode.error_resource_busy)
        if busy:
            message = ("The '{}' instrument is busy. If another program is using it, try "
                       "closing it or moving the instrument to a non-serial connection")
            log.error(message.format(instrument))
        else:
            message = "Error loading the '{}' instrument: " + str(error)
            log.error(message.format(instrument))
    except Exception as error:
        message = "Error loading the '{}' instrument: " + str(error)
        log.error(message.format(instrument))
    

class DAQ:
    """
    Data Acquisition System:
    Class for holding all of the instrument objects defined in the configuration
    dictionary. Also holds methods to initialize, close and reset all instruments at once.
    Data taking methods are selected from the run() method.
    """
    def __init__(self, configuration=DEFAULT_CONFIG):
        # load configuration
        if isinstance(configuration, str):
            file_name = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                     "configurations", configuration.lower() + ".yaml")
            with open(file_name) as f:
                self.config = yaml.load(f, Loader=yaml.FullLoader)
        else:
            self.config = configuration

        # initialize all instruments as None
        self.instrument_names = ('dac_atten',  # digital to analog converter
                                 'adc_atten',  # analog to digital converter
                                 'dac',  # dac attenuator
                                 'adc',  # adc attenuator
                                 'thermometer',  # device thermometer
                                 'primary_amplifier',  # first amplifier in the chain
                                 'laser')  # laser source
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
        for key, value in self.config['sources'].items():
            if key == "laser":
                self.laser = get_instrument(value)
        # if the instrument wasn't initialized set it to a dummy NotAnInstrument class
        if self.dac_atten is None:
            self.dac_atten = NotAnAttenuator("DAC")
        if self.adc_atten is None:
            self.adc_atten = NotAnAttenuator("ADC")
        if self.thermometer is None:
            self.thermometer = NotAThermometer()
        if self.primary_amplifier is None:
            self.primary_amplifier = NotASensor("Primary Amplifier")
        if self.laser is None:
            self.laser = NotASource("Laser Box")

        # Thermometer should be initialized immediately since it runs in the background
        self.thermometer.initialize()

        # set a flag that tracks if the all the instruments have been closed
        self.closed = False
        
    def procedure_class(self, procedure_type):
        """
        Return the procedure class of a given type.
        Args:
            procedure_type: sweep, noise, or pulse (str)
        """
        library = importlib.import_module("analogreadout.procedures")
        procedure_class = getattr(library, self.config["procedures"][procedure_type])
        procedure_class.connect_daq(self)
        return procedure_class
    
    def run(self, procedure_type, file_name_kwargs=None, should_stop=None, emit=None, indicators=None, **kwargs):
        """
        Take data for the given procedure_type. The procedure class is defined in the
        configuration file.
        Args:
        procedure_type: sweep, noise, or pulse (str)
        file_name_kwargs: kwargs to pass to procedure.file_name() after instantiation
        should_stop: method to monkey patch into procedure.stop  (for chained procedures)
        emit: method to monkey patch into procedure.emit (for sending data to listener)
        indicators: dictionary of indicators {"attribute name": indicator} to monkey patch into the procedure
        **kwargs: procedure parameters (set to the defaults if not specified)
        """
        if file_name_kwargs is None:
            file_name_kwargs = {}
        # get procedure class
        procedure = self.procedure_class(procedure_type)
        # overload the default parameter value and set it's value
        for key, value in kwargs.items():
            getattr(procedure, key).default = value
            getattr(procedure, key).value = value
        # check that all parameters have a default
        for name in dir(procedure):
            parameter = getattr(procedure, name)
            if isinstance(parameter, Parameter):
                message = "{} is not an optional parameter. No default is specified"
                assert parameter.default is not None, message.format(name)
        # run procedure
        procedure = procedure()
        if should_stop is not None:
            procedure.should_stop = should_stop
        if emit is not None:
            procedure.emit = emit
        if indicators is not None:
            for attribute, indicator in indicators.items():
                setattr(procedure, attribute, indicator)
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
        except AttributeError:
            file_name = None
        return file_name   

    def initialize(self, frequency, power=None, dac_atten=0, adc_atten=0,
                   sample_rate=None, n_samples=None, n_trace=None, channels=None, laser_state=None):
        """
        Initialize all of the instruments according to their initialize methods.
        Args:
            frequency: frequency to output from the DAC [GHz]
            power: power to output from the DAC [dBm] (optional, defaults to config value)
            dac_atten: DAC attenuation [dB] (optional, defaults to 0)
            adc_atten: ADC attenuation [dB] (optional, defaults to 0)
            sample_rate: ADC sample rate [Hz] (optional, default depends on ADC)
                The sample rate may be variable or only able to take one value depending
                on the hardware.
            n_samples: samples per ADC acquisition (optional, default depends on ADC)
            n_trace: samples per ADC acquisition trace (optional, typically used for pulse data)
            channels: ADC channels to take data with (optional, default depends on ADC)
            laser_state: state of the laser to initialize (optional, default depends on laser)
        """
        self.dac_atten.initialize(dac_atten)
        self.adc_atten.initialize(adc_atten)
        if power is None:
            power = self.config['dac']['dac']['power']
        self.dac.initialize(frequency, power)
        self.adc.initialize(sample_rate=sample_rate, n_samples=n_samples, channels=channels, n_trace=n_trace)
        self.thermometer.initialize()
        self.primary_amplifier.initialize()
        self.laser.initialize(laser_state)
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
            except Exception:
                problem = True
                message = "The '{}' instrument was unable to close: "
                log.error(message.format(instrument.__class__.__name__), exc_info=True)
        if not problem:
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
                log.warning(message.format(instrument.__class__.__name__) + str(error))
        
    def system_state(self):
        """
        Returns a dictionary defining the system state with a timestamp
        """
        temperatures = []
        resistances = []
        for _ in range(10):
            temperatures.append(self.thermometer.temperature)
            resistances.append(self.thermometer.resistance)
        
        thermometer = {'channel': self.thermometer.channel,
                       'temperature': (np.mean(temperatures), np.std(temperatures)),
                       'resistance': (np.mean(resistances), np.std(resistances))}
            
        state = {datetime.now().strftime('%Y%m%d_%H%M%S'):
                 {"thermometer": thermometer, "primary_amplifier": self.primary_amplifier.read_value()}}
        return state
