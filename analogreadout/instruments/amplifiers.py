import logging
import lakeshore
import numpy as np
from time import sleep

from analogreadout.instruments.signal_generators import AnritsuMG37022A

log = logging.getLogger(__name__)
log.addHandler(logging.NullHandler())


class ParaAmpThreeWaveUCSB:
    CONTROL = [["Reset", "reset", []],
               ["Recover", "recover_from_normal", []],
               ["Pump on", "turn_on_pump", []],
               ["Pump off", "turn_off_pump", []],
               ["Bias on", "turn_on_bias", []],
               ["Bias off", "turn_off_bias", []],
               ["Set Frequency", "set_pump_frequency",
                [[float, "", " GHz", 9]]],
               ["Set Power", "set_pump_power",
                [[float, "", " dBm", 3, 30, -20]]],
               ["Set Current", "set_bias_current", [[float, "", " mA", 3]]]]

    def __init__(self, pump_address, bias_address, bias_resistor=1e3):
        self.pump = AnritsuMG37022A(pump_address)
        self.bias = lakeshore.Model155(com_port=bias_address)
        self.bias_resistor = float(bias_resistor)

        state = self.get_state()
        self.pump_power = state['pump']['power']
        self.bias_current = state['bias']['amplitude']

        bias_name = "{:s} {:s}, s/n: {:s}, version: {:s}".format(
            *self.bias.query("*IDN?").strip().split(','))
        pump_name = "{:s} {:s}, s/n: {:s}, version: {:s}".format(
            *self.pump.query("*IDN?").strip().split(',')
        )
        log.info("Connected to UCSB para-amp controller. "
                 f"Pump: {pump_name} :: Bias: {bias_name}")

    def initialize(self):
        # turn everything on to its lowest power setting
        self.turn_off_bias()
        self.turn_off_pump()
        self.pump_power = -20
        self.bias_current = 0
        self.pump.set_power(self.pump_power)
        self.bias.command("SOURCE:FUNCTION:MODE CURRENT")
        self.bias.command("SOURCE:FUNCTION:SHAPE DC")
        self.bias.command("SOURCE:CURRENT:AMPLITUDE " + str(self.bias_current))
        self.turn_on_bias()
        self.turn_on_pump()

    def get_state(self):
        bias_settings = self.bias.get_output_settings()
        bias_settings['output_state'] = bool(int(
            self.bias.query("OUTPut:STATe?")))
        bias_settings['terminals'] = self.bias.query("ROUTe:TERMinals?")
        bias_settings['amplitude'] *= 1e3  # to mA
        return {'bias': bias_settings, 'pump': self.pump.get_state()}

    def close(self):
        self.pump.close()
        self.bias.disconnect_usb()

    def reset(self):
        self.pump.reset()
        self.bias.command("*RST")

    def turn_on_pump(self):
        self.pump.turn_on_output()

    def turn_off_pump(self):
        self.pump.turn_off_output()

    def set_pump_power(self, power):
        sleep_time = 0.1
        # take 2 minutes to change the power by 30 dB
        n_steps = int(120 / 30 * (power - self.pump_power) / sleep_time)
        x = np.linspace(0, 1, n_steps)
        powers = x * (power - self.pump_power) + self.pump_power
        for p in powers:  # slow ramp
            self.pump.set_power(p)
            sleep(sleep_time)
            if self.is_normal():
                self.recover_from_normal()
                break
        else:
            self.pump_power = power  # did not go normal

    def set_pump_frequency(self, frequency):
        state = self.get_state()

        # If the pump frequency is changed with the pump and bias powered,
        # the para-amp might go normal, so we turn everything off.
        self.turn_off_pump()
        self.turn_off_bias()

        self.pump.set_frequency(frequency)

        # return to previous state
        if state['bias']['output_state']:
            self.turn_on_bias()
        if state['pump']['output_state']:
            self.turn_on_pump()
            # check if we went normal if we turned the pump on.
            sleep(1)
            if self.is_normal():
                self.recover_from_normal()
                log.warning("Para-amp went normal")

    def turn_on_bias(self):
        self.bias.enable_output()

    def turn_off_bias(self):
        self.bias.disable_output()

    def set_bias_current(self, current):
        ranges = ["100e-3", "10e-3", "1e-3", "100e-6", "10e-6", "1e-6"]
        difference = np.array([float(r) for r in ranges]) - current * 1e-3
        current_range = ranges[np.argmin(difference[difference > 0])]
        self.bias.set_current_range(current_range)
        # Limit the voltage to no more than 1% of the required value to reduce
        # unintentional para-amp heating.
        volts = min(max(current * 1e-3 * self.bias_resistor * 1.01, 1), 100)
        self.bias.set_current_mode_voltage_protection(volts)
        # The default lakeshore method turns the output on, but we don't want
        # to do that, so we send the relevant commands directly.
        self.bias.command("SOURCE:FUNCTION:MODE CURRENT")
        self.bias.command("SOURCE:FUNCTION:SHAPE DC")
        self.bias.command("SOURCE:CURRENT:AMPLITUDE " + str(current * 1e-3))
        self.bias_current = current

    def is_normal(self):
        state = self.bias.query(
            "SOURce:CURRent:SENSe:VOLTage:DC:PROTection:TRIPped?")
        return bool(int(state))

    def recover_from_normal(self):
        self.turn_off_pump()
        self.turn_off_bias()
        self.pump.set_power(-20)
        self.pump_power = -20


class NotAnAmplifier:
    def __init__(self, name=''):
        self.name = name
        message = "The {} is not connected and was not initialized"
        log.warning(message.format(self.name))

    def initialize(self):
        pass

    @staticmethod
    def get_state():
        return np.nan

    def close(self):
        pass

    def reset(self):
        pass
