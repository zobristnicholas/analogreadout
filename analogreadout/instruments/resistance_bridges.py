import visa
import logging
import numpy as np
from time import sleep
from slave.types import Integer
from scipy.interpolate import interp1d
from slave.lakeshore.ls370 import LS370, Heater
from slave.transport import Visa, SimulatedTransport

log = logging.getLogger(__name__)
log.addHandler(logging.NullHandler())

# patch Integer type in slaves module
Integer.__convert__ = lambda self, value: int(float(value))


class LakeShore370AC(LS370):
    WAIT_MEASURE = 0.15

    def __init__(self, address, channel, scanner=None):
        self.channel = channel
        transport = Visa(address)            
        # change data format for RS-232 serial control
        transport._instrument.parity = visa.constants.Parity.odd
        transport._instrument.data_bits = 7
        super().__init__(transport, scanner=scanner)
        identity = self.identification
        self.identity = [s.strip() for s in identity]
        log.info("Connected to: %s %s, s/n: %s, version: %s", *self.identity)

        temperatures = [-1, 9.1, 11.4, 16.2, 22, 28.2, 34.3, 40.6, 46.7, 53, 59.3, 65.5, 71.5, 78, 84, 90, 96, 104]
        percentages = [0, 0, 4, 7, 10, 13, 16, 19, 22, 25, 28, 31, 34, 37, 40, 43, 47, 50]
        self.calibration = interp1d(temperatures, percentages)
        # patch for missing heater        
        self.heater = Heater(self._transport, self._protocol)
        # keep track of set point so that you don't have to keep waiting if the temperature isn't changing
        self._set_point = None
        
    def initialize(self):
        # set channel and turn autoscan off
        self.input.scan = self.channel, False
        sleep(self.WAIT_MEASURE)
        # lowest bias setting (may be modified to a different setting later)
        self.set_bias(2, 'voltage')
            
    @property
    def temperature(self):
        temp = self.input[self.channel - 1].kelvin
        sleep(self.WAIT_MEASURE)
        return temp

    
    @property
    def resistance(self):
        res = self.input[self.channel - 1].resistance
        sleep(self.WAIT_MEASURE)
        return res
        
    def set_temperature(self, temperature, heater_range=5, max_wait=60, min_wait=10):
        if max_wait <= 0 or self.calibration(temperature) <= 0 or self._set_point == temperature:
            return
        log.debug("Setting temperature to {} mK".format(temperature))
        self._set_point = temperature
        self.set_range(heater_range)
        if temperature < 50:
            self.set_bias(2, 'voltage')
        else:
            self.set_bias(3, 'voltage')
        previous_temperature = 0
        n_sleep, n_eq = 0, 0
        self.set_heater_output(self.calibration(temperature))
        while n_sleep < max_wait and n_eq < min_wait:
            temperatures = []
            for _ in range(10):
                t = self.temperature * 1000
                temperatures.append(t)
            current_temperature = np.mean(temperatures)
            log.info("Current temperature: {:.2f} +/- {:.2f} mK".format(current_temperature, np.std(temperatures)))
            deviation = np.abs(previous_temperature - current_temperature)

            if deviation < 0.5 * np.std(temperatures):
                n_eq += 1
            else:
                previous_temperature = current_temperature
            n_sleep += 1
            sleep(60)
                
    def set_range(self, heater_range=5):
        self.heater.range = Heater.RANGE[heater_range] 
        sleep(self.WAIT_MEASURE)
        
    def set_heater_output(self, level):
        self.heater.manual_output = level
        sleep(self.WAIT_MEASURE)
        
    def set_bias(self, index, mode='voltage', auto_range=True):
        """
        Voltage Mode    Current Mode
        1 2.00 μV       1 1.00 pA
        2 6.32 μV       2 3.16 pA
        3 20.0 μV       3 10.0 pA
        4 63.2 μV       4 31.6 pA
        5 200 μV        5 100 pA
        6 632 μV        6 316 pA
        7 2.00 mV       7 1.00 nA
        8 6.32 mV       8 3.16 nA
        *9 20.0 mV      9 10.0 nA
        *10 63.2 mV     10 31.6 nA
        *11 200 mV      11 100 nA
        *12 632 mV      12 316 nA
                        13 1.00 μA
                        14 3.16 μA
                        15 10.0 μA
                        16 31.6 μA
                        17 100 μA
                        18 316 μA
                        19 1.00 mA
                        20 3.16 mA
                        21 10.0 mA
                        22 31.6 mA
        """
        settings = self.input[self.channel - 1].resistance_range
        r_range = settings[2]
        self.input[self.channel - 1].resistance_range = mode, index, r_range, auto_range, False
        sleep(self.WAIT_MEASURE)
        
    def close(self):
        try:
            self.heater.range = Heater.RANGE[0]
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
        log.warning("Thermometer does not exist and will be ignored")
        self.channel = np.nan
        self.temperature = np.nan
        self.resistance = np.nan
        
    def initialize(self):
        pass

    def set_temperature(self, temperature, heater_range=5, max_wait=60, min_wait=10):
        pass
    
    def set_range(self, heater_range=5):
        pass
        
    def set_heater_output(self, level):
        pass
        
    def close(self):
        pass
    
    def reset(self):
        pass
