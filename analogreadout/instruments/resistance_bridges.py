import visa
import logging
import numpy as np
from time import sleep
from slave.types import Integer
from scipy.interpolate import interp1d
from slave.lakeshore.ls370 import LS370
from slave.transport import Visa, SimulatedTransport

log = logging.getLogger(__name__)
log.addHandler(logging.NullHandler())

# patch Integer type in slaves module
Integer.__convert__ = lambda self, value: int(float(value))


class LakeShore370AC(LS370):
    WAIT_MEASURE = 0.1

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
        
    def initialize(self):
        pass
            
    @property
    def temperature(self):
        return self.input[self.channel - 1].kelvin
    
    @property
    def resistance(self):
        return self.input[self.channel - 1].resistance
        
    def set_temperature(self, temperature, heater_range='3.16 mA', max_wait=60):
        log.info("Setting temperature to {} K".format(temperature))
        self.heater.range(heater_range)
        previous_temperature = 0
        n_sleep = 0
        while n_sleep < max_wait:
            self.heater.manual_output(self.calibration(temperature))
            temperatures = []
            for _ in range(10):
                temperatures.append(self.temperature)
                sleep(self.WAIT_MEASURE)
            current_temperature = np.mean(temperatures)
            log.info("Current temperature: {} K".format(current_temperature))
            deviation = np.abs(previous_temperature - current_temperature)

            if deviation < 0.5 * np.std(temperatures):
                break
            else:
                previous_temperature = current_temperature
                n_sleep += 1
                sleep(60)
        
    def close(self):
        self.heater.range('off')
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

    def set_temperature(self, temperature):
        pass
        
    def close(self):
        pass
    
    def reset(self):
        pass
