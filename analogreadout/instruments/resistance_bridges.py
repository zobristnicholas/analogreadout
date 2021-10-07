import os
import yaml
import pyvisa
import logging
import threading
import numpy as np
from time import sleep
from pyvisa.constants import Parity
from scipy.interpolate import interp1d
from analogreadout.external.lakeshore370ac.ls370 import LS370
from analogreadout.external.lakeshore370ac.transport import Visa, SimulatedTransport

log = logging.getLogger(__name__)
log.addHandler(logging.NullHandler())
temperature_log = logging.getLogger('temperature')
temperature_log.addHandler(logging.NullHandler())
resistance_log = logging.getLogger('resistance')
resistance_log.addHandler(logging.NullHandler())


LOCK = threading.Lock()


class LakeShore370AC(LS370):
    WAIT_MEASURE = 0.15
    WAIT_MEASURE_LONG = 1

    def __init__(self, address, thermometer, scanner=None):
        if not os.path.isfile(thermometer):
            file_name = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..",
                                     "configurations", thermometer.lower() + ".yaml")
        else:
            file_name = thermometer
        with open(file_name, "r") as f:
            self.config = yaml.load(f, Loader=yaml.FullLoader)
        self.channel = self.config['channel']
        transport = Visa(address, parity=Parity.odd, data_bits=7)
        super().__init__(transport, scanner=scanner)
        identity = self.identification
        self.identity = [s.strip() for s in identity]
        log.info("Connected to: %s %s, s/n: %s, version: %s", *self.identity)

        temperatures = [-1, 9.1, 11.4, 16.2, 22, 28.2, 34.3, 40.6, 46.7, 53, 59.3, 65.5, 71.5, 78, 84, 90, 96, 104]
        percentages = [0, 0, 4, 7, 10, 13, 16, 19, 22, 25, 28, 31, 34, 37, 40, 43, 47, 50]
        self.calibration = interp1d(temperatures, percentages, bounds_error=False, fill_value='extrapolate')
        # keep track of set point so that you don't have to keep waiting if the temperature isn't changing
        self._set_point = None
        
    def initialize(self):
        # set channel and turn autoscan off
        self.input.scan = self.channel, False
        sleep(self.WAIT_MEASURE)
        self.set_bias()
            
    @property
    def temperature(self):
        with LOCK:
            try:
                sleep(self.WAIT_MEASURE)
                temp = self.input[self.channel - 1].kelvin
                temperature_log.info(f"{temp * 1000:g} mK")
            except pyvisa.VisaIOError:
                log.error("The {:s} {:s}, s/n: {:s}, version: {:s} ".format(*self.identity)
                          + "failed to measure the temperature.", exc_info=True)
                temp = np.nan
        return temp
    
    @property
    def resistance(self):
        with LOCK:
            try:
                sleep(self.WAIT_MEASURE)
                res = self.input[self.channel - 1].resistance
                resistance_log.info(f"{res:g} Ohm")
            except pyvisa.VisaIOError:
                log.error("The {:s} {:s}, s/n: {:s}, version: {:s} ".format(*self.identity)
                          + "failed to measure the resistance.", exc_info=True)
                res = np.nan
        return res
        
    def set_temperature(self, temperature, heater_range=5, wait=30, stop=None):
        if wait <= 0 or self.calibration(temperature) <= 0 or self._set_point == temperature:
            return
        log.debug("Setting temperature to {} mK".format(temperature))
        self._set_point = temperature
        self.set_heater_range(heater_range)
        self.set_bias()  # uses self._set_point and self.config to determine the bias parameters
        n_sleep = 0
        self.set_heater_output(self.calibration(temperature))
        while n_sleep < wait:
            temperatures = []
            for _ in range(10):
                t = self.temperature * 1000
                temperatures.append(t)
            current_temperature = np.mean(temperatures)
            log.info("Current temperature: {:.2f} +/- {:.2f} mK".format(current_temperature, np.std(temperatures)))
            n_sleep += 1
            for ii in range(20):  # sleep for 60 seconds while checking if we are stopping
                sleep(3)
                if callable(stop) and stop():
                    return
                
    def set_heater_range(self, heater_range=5):
        self.heater.range = self.heater.RANGE[heater_range]
        sleep(self.WAIT_MEASURE)
        
    def set_heater_output(self, level):
        self.heater.manual_output = level
        sleep(self.WAIT_MEASURE)
        
    def set_bias(self, **kwargs):
        """
        Voltage Mode    Current Mode    Resistance Range
        1 2.00 μV       1 1.00 pA       1  2.00 mOhm
        2 6.32 μV       2 3.16 pA       2  6.32 mOhm
        3 20.0 μV       3 10.0 pA       3  20.0 mOhm
        4 63.2 μV       4 31.6 pA       4  63.2 mOhm
        5 200 μV        5 100 pA        5  200 mOhm
        6 632 μV        6 316 pA        6  632 mOhm
        7 2.00 mV       7 1.00 nA       7  2.00 Ohm
        8 6.32 mV       8 3.16 nA       8  6.32 Ohm
        *9 20.0 mV      9 10.0 nA       9  20.0 Ohm
        *10 63.2 mV     10 31.6 nA      10 63.2 Ohm
        *11 200 mV      11 100 nA       11 200 Ohm
        *12 632 mV      12 316 nA       12 632 Ohm
                        13 1.00 μA      13 2.00 kOhm
                        14 3.16 μA      14 6.32 kOhm
                        15 10.0 μA      15 20.0 kOhm
                        16 31.6 μA      16 63.2 kOhm
                        17 100 μA       17 200 kOhm
                        18 316 μA       18 632 kOhm
                        19 1.00 mA      19 2.00 MOhm
                        20 3.16 mA      20 6.32 MOhm
                        21 10.0 mA      21 20.0 MOhm
                        22 31.6 mA      22 63.2 MOhm
        """
        # Get the bias settings from the config file based on the set point
        for temperature_range in self.config['ranges']:
            if self._set_point is None:
                continue  # go to else
            if temperature_range['start'] <= self._set_point < temperature_range['stop']:
                settings = temperature_range['bias']
                break
        else:  # didn't find a valid temperature range so we use the lowest start temperature
            settings = min(self.config['ranges'], key=lambda x: x['start'])['bias'].copy()

        # Override the settings with any provided kwargs.
        settings.update(kwargs)

        # Set default values for some of the settings.
        if settings.get('resistance', None) is None:
            resistance_range = self.input[self.channel - 1].resistance_range
            settings['resistance'] = resistance_range[2]
        if settings.get('mode', None) is None:
            settings['mode'] = 'voltage'
        if settings.get('autorange', None) is None:
            settings['autorange'] = True

        # Send the command to the device.
        command = (settings['mode'], settings['index'], settings['resistance'], settings['autorange'], False)
        self.input[self.channel - 1].resistance_range = command
        sleep(self.WAIT_MEASURE_LONG)
        
    def close(self):
        try:
            self.heater.range = self.heater.RANGE[0]
        except NameError:
            pass
        self._transport._instrument.close()
        message = "The visa session for {} {}, s/n: {} has been closed"
        log.info(message.format(*self.identity[:3]))
        
    def __del__(self):
        # extra insurance that the serial port doesn't lock up if something goes wrong
        try:
            self.close()
        except AttributeError:
            pass


class NotAThermometer(LS370):
    WAIT_MEASURE = 0

    def __init__(self):
        transport = SimulatedTransport()            
        super().__init__(transport)
        log.warning("No thermometer was connected.")
        self.channel = np.nan
        self.temperature = np.nan
        self.resistance = np.nan
        
    def initialize(self):
        pass

    def set_bias(self, **kwargs):
        pass

    def set_temperature(self, temperature, heater_range=5, wait=0, stop=None):
        pass
    
    def set_heater_range(self, heater_range=5):
        pass
        
    def set_heater_output(self, level):
        pass
        
    def close(self):
        pass
    
    def reset(self):
        pass
