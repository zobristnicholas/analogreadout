import visa
import warnings
import numpy as np
from time import sleep
from analogreadout.custom_warnings import ConnectionWarning

    
class Weinschel83102042F:
    MAX_ATTEN = 62
    STEP = 2
    def __init__(self, address, channels):
        try:
            resource_manager = visa.ResourceManager()
        except:
            resource_manager = visa.ResourceManager('@py')
        self.session = resource_manager.open_resource(address)
        identity = self.query_ascii_values("*IDN?", 's')
        print("Connected to:", identity[0])
        print("Model Number:", identity[1])
        print("Serial Number:", identity[2])
        print("System Version:", identity[3])
        self.channels = channels  # list of chained channels to get more attenuation range
        
    def initialize(self, attenuation):
        self.reset()
        self.set_attenuation(attenuation)

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
        sleep(2)
    
    def _write_attenuation(self, attenuation):
        self.write("ATTN {:d}".format(attenuation))
        
    def _write_channel(self, channel):
        self.write("CHAN {:d}".format(channel))
    
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

    def reset(self):
        self.write("*RST")
        sleep(5)


class NotAnAttenuator:
    def __init__(self, name=''):
        self.name = name
        self.warn("connection")

    def initialize(self, attenuation):
        if attenuation != 0:
            self.warn("set")

    def set_attenuation(self, attenuation):
        if attenuation != 0:
            self.warn("set")

    def close(self):
        pass

    def reset(self):
        pass

    def warn(self, warning_type):
        if warning_type == "connection":
            message = "{} attenuator does not exist and will be ignored"
            warnings.warn(message.format(self.name), ConnectionWarning)
        elif warning_type == "set":
            message = "{} attenuator does not exist so it can not be set"
            warnings.warn(message.format(self.name), UserWarning)
