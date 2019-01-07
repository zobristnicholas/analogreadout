import visa
import numpy as np
from time import sleep


class MultipleSignalGenerators(list):
    def __init__(self, classes, arguments):
        super().__init__()
        # make a list of classes if only one class was given
        if len(classes) != len(arguments) and len(classes) == 1:
            classes = list(classes) * len(arguments)
        # initiate and store the signal generators in a list
        for index, argument in enumerate(arguments):
            self.append(globals()[classes[index]](*argument))

    def __getattr__(self, name):
        # if it's an attribute return a list of attributes
        if not callable(getattr(self[0], name)):
            return [getattr(self[0], name), getattr(self[1], name)]

        # if not, return a list of the method outputs for each object
        def new_method(*args, **kwargs):
            container = []
            for index, sig_gen in enumerate(self):
                # get the method from the signal generator in the list
                old_method = getattr(sig_gen, name)

                # make sure that each argument is a list of arguments
                args = list(args)
                for ind, arg in enumerate(args):
                    if not isinstance(arg, (list, tuple, np.ndarray)):
                        args[ind] = [arg] * len(self)
                for key, value in kwargs.items():
                    if not isinstance(value, (list, tuple, np.ndarray)):
                        kwargs[key] = [value] * len(self)
                # modify the args
                new_args = [arg[index] for arg in args]
                new_kwargs = {key: value[index] for key, value in kwargs.items()}
                # run the method and put the result in the container
                container.append(old_method(*new_args, **new_kwargs))
            # return None if no output from old_method
            if all(element is None for element in container):
                return None
            return container
        return new_method


class AnritsuABC:
    def __init__(self, address, power):
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
        self.power = power
        self.set_power(power)

    def initialize(self, frequency, power=None):
        self.turn_off_output()
        self.reset()
        self.set_frequency(frequency)
        if power is None:
            power = self.power
        self.set_power(power)
        self.turn_on_output()
        sleep(1)
        
    def set_frequency(self, frequency):
        if isinstance(frequency, (list, tuple, np.ndarray)):
            if len(frequency) != 1:
                raise ValueError("can only set one frequency at a time")
            frequency = frequency[0]
        self.write("F1 {} GH;".format(frequency))
        sleep(0.05)

    def set_power(self, power):
        if isinstance(power, (list, tuple, np.ndarray)):
            if len(power) != 1:
                raise ValueError("can only set one power at a time")
            power = power[0]
        self.write("L1 {} DM;".format(power))
        sleep(0.5)
        
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
        
    def turn_on_output(self):
        raise NotImplementedError
        
    def turn_off_output(self):
        raise NotImplementedError


class AnritsuMG37022A(AnritsuABC):
    def turn_on_output(self):
        self.write("OUTPut: ON")
        sleep(0.5)

    def turn_off_output(self):
        self.write("OUTPut: OFF")
        sleep(0.5)


class AnritsuMG3692B(AnritsuABC):
    def set_increment(self, frequency):
        if isinstance(frequency, (list, tuple, np.ndarray)):
            if len(frequency) != 1:
                raise ValueError("can only set one frequency at a time")
            frequency = frequency[0]
        self.write("F1 SYZ {:f} GH".format(frequency))
        sleep(0.5)
    
    def increment(self):
        self.write("F1 UP")
        sleep(0.1)

    def turn_on_output(self):
        self.write("RF1")
        sleep(0.5)

    def turn_off_output(self):
        self.write("RF0")
        sleep(0.5)

