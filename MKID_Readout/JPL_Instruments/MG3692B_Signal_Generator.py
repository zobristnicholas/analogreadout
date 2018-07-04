import visa


class MG3692B_Signal_Generator:
    def __init__(self, address):
        try:
            resourceManager = visa.ResourceManager()
        except:
            resourceManager = visa.ResourceManager('@py')
        self.session = resourceManager.open_resource(address)
        print(self.session.query("*IDN?"))

    def initialize(self, frequency, power):
        self.set_frequency(frequency)
        self.set_power(power)

    def set_frequency(self, frequency):
        self.session.write(":FREQUENCY {:e} Hz;".format(frequency))

    def set_power(self, power):
        self.session.write(":POWER {:g} dBm;".format(power))
