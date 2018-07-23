import visa
from time import sleep


class Programmable_Attenuator_83102042F:
    def __init__(self, address):
        try:
            resourceManager = visa.ResourceManager()
        except:
            resourceManager = visa.ResourceManager('@py')
        self.session = resourceManager.open_resource(address)
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
        if attenuation > 62:
            print("62 dB is the max attenuation")
            print("setting at 62 dB")
            self.write("ATTN 62")
        elif attenuation < 0:
            print ("0 dB is the minimum attenuation")
            print("setting at 0 dB")
            self.write("ATTN 0")
        elif (attenuation % 2) != 0:
            print("Only attenuations divisible by 2 allowed")
            print("setting at {:.0f} dB".format(attenuation - attenuation % 2))
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