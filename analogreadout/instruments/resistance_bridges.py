import visa
import logging
import numpy as np
from slave.types import Integer
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
        
    def initialize(self):
        pass
            
    @property
    def temperature(self):
        return self.input[self.channel - 1].kelvin
    
    @property
    def resistance(self):
        return self.input[self.channel - 1].resistance
        
    def set_temperature(self, temperature):
        # TODO: implement some logic to set the temperature
        pass
        
    def close(self):
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
        passs
        
    def set_temperature(self, temperature):
        pass
        
    def close(self):
        pass
    
    def reset(self):
        pass
        
        
    
