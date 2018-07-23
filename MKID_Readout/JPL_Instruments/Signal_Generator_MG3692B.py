import visa
from time import sleep


class Signal_Generator_MG3692B:
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


    def initialize(self, frequency, power):
        self.reset()
        self.turn_on_output()
        self.set_frequency(frequency)
        self.set_power(power)
        sleep(1)

    def set_frequency(self, frequency):
        self.write("F1 {} GH;".format(frequency))
        sleep(0.5)

    def set_power(self, power):
        self.write("L1 {} DM;".format(power))
        sleep(0.5)
        
    def set_increment(self, frequency):
        self.write("F1 SYZ {:f} GH".format(frequency))
        sleep(0.5)
        
    def turn_on_output(self):
        self.write("RF1")
        sleep(0.5)
    
    def turn_off_output(self):
        self.write("RF0")
        sleep(0.5)
        
    def increment(self):
        self.write("F1 UP")
        sleep(0.02)

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
        self.turn_off_output()
        self.session.close()

    def reset(self):
        self.write("*RST")
        sleep(1)