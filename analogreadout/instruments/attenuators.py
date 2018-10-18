import visa
import warnings
import numpy as np
from time import sleep
from analogreadout.custom_warnings import ConnectionWarning


class Weinschel83102042F:
    def __init__(self, address):
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

    def initialize(self, attenuation):
        self.reset()
        self.set_attenuation(attenuation)

    def set_attenuation(self, attenuation):
        self.write("CHAN 2")
        self.write("ATTN 0")
        self.write("CHAN 1")
        if np.isinf(attenuation) and attenuation > 0:
            self.write("CHAN 2")
            self.write("ATTN 62")
            self.write("CHAN 1")
            self.write("ATTN 62")
        elif attenuation > 62:
            warnings.warn("setting at 62 dB, the max attenuation", UserWarning)
            self.write("ATTN 62")
        elif attenuation < 0:
            warnings.warn("setting at 0 dB, the min attenuation", UserWarning)
            self.write("ATTN 0")
        elif (attenuation % 2) != 0:
            message = "setting at {:.0f} dB, only attenuations divisible by 2 are allowed"
            warnings.warn(message.format(attenuation - attenuation % 2))
            self.write("ATTN {}".format(attenuation - attenuation % 2))
        else:
            self.write("ATTN {}".format(attenuation))
        sleep(2)

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
