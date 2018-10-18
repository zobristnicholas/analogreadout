import warnings
from analogreadout.daq import DAQ
from analogreadout.configurations import config
from analogreadout.custom_warnings import ConnectionWarning
from analogreadout.functions import take_noise_data, do_iq_sweep, take_pulse_data

DEFAULT_CONFIG = "JPL"


class System:
    """
    Class for holding a DAQ object and the methods which act on it
    """
    def __init__(self, configuration=None):
        if configuration is None:
            self.config = config(DEFAULT_CONFIG)
        else:
            self.config = config(configuration)
        self.daq = DAQ(self.config)

        # only display missing instrument warnings once
        warnings.simplefilter("once", ConnectionWarning)

    def take_noise_data(self, *args, **kwargs):
        """
        Take noise data
        Args:
            frequency: frequency [GHz]
            dac_atten: dac attenuation [dB]
            n_triggers: number of noise triggers (int)
            directory: folder where data should be saved (string)
            power: dac power [dB] (optional, should be set by configuration)
            adc_atten: adc attenuation [dB] (optional, defaults to 0)
            sample_rate: sample rate of adc (float, defaults to 2e6 Hz)
            verbose: print information about the system (bool, defaults to True)
        
        Returns:
            file_path: full path where the data was saved
        """
        if "power" not in kwargs.keys():
            kwargs.update({"power": self.config['dac']['dac']['power']})
        return take_noise_data(self.daq, *args, **kwargs)

    def do_iq_sweep(self, *args, **kwargs):
        """
        Take an iq sweep
        Args:
            center: center frequency [GHz]
            span: sweep span [GHz]
            dac_atten: dac attenuation [dB]
            n_points: number of points in the sweep (int)
            directory: folder where data should be saved (string)
            power: dac power [dB] (optional, should be set by configuration)
            adc_atten: adc attenuation [dB] (optional, defaults to 0)
            verbose: print information about the system (bool, defaults to True)
            
        Returns:
            file_path: full path where the data was saved.
        """
        if "power" not in kwargs.keys():
            kwargs.update({"power": self.config['dac']['dac']['power']})
        return do_iq_sweep(self.daq, *args, **kwargs)

    def take_pulse_data(self, *args, **kwargs):
        """
        Take pulse data
        Args:
            frequency: frequency [GHz]
            dac_atten: dac attenuation [dB]
            n_triggers: number of noise triggers (int)
            directory: folder where data should be saved (string)
            power: dac power [dB] (optional, should be set by configuration)
            adc_atten: adc attenuation [dB] (optional, defaults to 0)
            sample_rate: sample rate of adc (float, defaults to 2e6 Hz)
            verbose: print information about the system (bool, defaults to True)
            
        Returns:
            file_path: full path where the data was saved
        """
        if "power" not in kwargs.keys():
            kwargs.update({"power": self.config['dac']['dac']['power']})
        return take_pulse_data(self.daq, *args, **kwargs)

    # def find_attenuations(self, power_at_device, passive_attenuation):
    #     adc_power = self.config['adc']['adc']['power']
    #     dac_attenuation = adc_power - passive_attenuation - power_at_device