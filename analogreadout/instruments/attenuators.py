import logging
import warnings
import numpy as np
import pyvisa as visa
from time import sleep

log = logging.getLogger(__name__)
log.addHandler(logging.NullHandler())

    
class Weinschel83102042F:
    MAX_ATTEN = 62
    STEP = 2
    CONTROL = [["Reset", "reset", []],
               ["Set Attenuation", "set_attenuation", [[float, "", " dB", 0, 124]]]]

    def __init__(self, address, channels):
        try:
            resource_manager = visa.ResourceManager()
        except:
            resource_manager = visa.ResourceManager('@py')
        self.session = resource_manager.open_resource(address)
        identity = self.query_ascii_values("*IDN?", 's')
        self.identity = [s.strip() for s in identity]
        log.info("Connected to: %s %s, s/n: %s, version: %s", *self.identity)
        self.channels = channels  # list of chained channels to get more attenuation range
        
    def initialize(self, attenuation):
        self.reset()
        self.set_attenuation(attenuation)
        sleep(5)

    def set_attenuation(self, attenuation):
        # check for step size compatibility
        if (attenuation % self.STEP) != 0 and not np.isinf(attenuation):
            new_attenuation = min(self.MAX_ATTEN * len(self.channels),
                                  attenuation - attenuation % self.STEP) 
            message = "setting at {:.0f} dB, must be divisible by {:d}"
            warnings.warn(message.format(new_attenuation, self.STEP))                        
            return self.set_attenuation(new_attenuation)             
        # set the attenuation
        if attenuation > self.MAX_ATTEN * len(self.channels):
            warnings.warn("setting at {} dB, the max attenuation"
                          .format(self.MAX_ATTEN * len(self.channels)), UserWarning)
            for channel in self.channels:
                self._write_channel(channel)
                self._write_attenuation(self.MAX_ATTEN)
        elif attenuation < 0:
            warnings.warn("setting at 0 dB, the min attenuation", UserWarning)
            for channel in self.channels:
                self._write_channel(channel)
                self._write_attenuation(0)
        else:
            current_attenuation = 0
            for index, channel in enumerate(self.channels):
                self._write_channel(channel)
                new_attenuation = min(self.MAX_ATTEN, attenuation - current_attenuation)
                self._write_attenuation(new_attenuation)
                current_attenuation += new_attenuation
        sleep(0.5)
    
    def _write_attenuation(self, attenuation):
        self.write("ATTN {:d}".format(int(attenuation)))
        
    def _write_channel(self, channel):
        self.write("CHAN {:d}".format(int(channel)))
    
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
    
    def close(self):
        self.session.close()
        message = "The visa session for {} {}, s/n: {} has been closed"
        log.info(message.format(*self.identity[:3]))

    def reset(self):
        self.write("*RST")
        sleep(1)


class NotAnAttenuator:
    def __init__(self, name=''):
        self.name = name
        message = "{} attenuator does not exist and will be ignored"
        log.warning(message.format(self.name))

    def initialize(self, attenuation):
        pass

    def set_attenuation(self, attenuation):
        pass

    def close(self):
        pass

    def reset(self):
        pass
