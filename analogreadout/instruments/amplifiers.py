import logging
import threading
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

    WAIT = 0.1

    def __init__(self, pump_address, bias_address, bias_resistor=1e3):
        self.pump = AnritsuMG37022A(pump_address)
        self.bias = lakeshore.Model155(com_port=bias_address)
        self.bias_resistor = float(bias_resistor)

        # Create a thread that monitors the pump to handle if it goes normal.
        self.normal = False  # flag set by thread if the pump goes normal
        self.thread = threading.Thread(target=self._monitor, daemon=True)
        self.thread.start()

        # Get the current pump state.
        state = self.get_state()
        self.pump_power = state['pump']['power']
        self.bias_current = state['bias']['amplitude']
        if state['bias']['mode'] != "CURRENT":
            self.bias.command("SOURCE:FUNCTION:MODE CURRENT")
        if state['bias']['output_shape'] != "DC":
            self.bias.command("SOURCE:FUNCTION:SHAPE DC")

        # Log the para-amp connection information.
        bias_name = "{:s} {:s}, s/n: {:s}, version: {:s}".format(
            *self.bias.query("*IDN?").strip().split(','))
        pump_name = "{:s} {:s}, s/n: {:s}, version: {:s}".format(
            *self.pump.query("*IDN?").strip().split(','))
        log.info("Connected to UCSB para-amp controller. "
                 f"Pump: {pump_name} :: Bias: {bias_name}")

    def _monitor(self):
        while True:
            sleep(1)
            if self.is_normal():
                self.recover_from_normal()
                self.normal = True
                log.warning("The para-amp went normal and was shut off "
                            "automatically.")
                log.debug("The para-amp normal flag was set.")
                sleep(10)
            self.normal = False
            log.debug("The para-amp normal flag was unset.")

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
        sleep(self.WAIT)

    def turn_on_pump(self):
        self.pump.turn_on_output()
        sleep(self.WAIT)

    def turn_off_pump(self):
        self.pump.turn_off_output()
        sleep(self.WAIT)

    def set_pump_power(self, power):
        sleep_time = 0.1
        # take 2 minutes to change the power by 30 dB
        n_steps = int(120 / 30 * np.abs(power - self.pump_power) / sleep_time)
        x = np.linspace(0, 1, n_steps)
        powers = x * (power - self.pump_power) + self.pump_power
        for p in powers:  # slow ramp
            self.pump.set_power(p)
            sleep(sleep_time)
            if self.is_normal():
                self.recover_from_normal()
                log.warning("The para-amp went normal while setting the pump "
                            "power and was shut off automatically.")
                break
        else:
            self.pump_power = power  # did not go normal

    def set_pump_frequency(self, frequency):
        # Get the state of the pump and bias power.
        state = self.get_state()
        on = state['pump']['output_state'] and state['bias']['output_state']

        # If the pump frequency is changed with the pump and bias powered,
        # the para-amp might go normal, so we turn the pump off.
        if on:
            self.turn_off_pump()

        # Set the frequency.
        self.pump.set_frequency(frequency)

        # Return to the previous state.
        if on:
            self.turn_on_pump()
            sleep(1)
            if self.is_normal():
                self.recover_from_normal()
                log.warning("The para-amp went normal while setting the pump "
                            "frequency and was shut off automatically.")

    def turn_on_bias(self):
        self.bias.enable_output()
        sleep(self.WAIT)

    def turn_off_bias(self):
        self.bias.disable_output()
        sleep(self.WAIT)

    def set_bias_current(self, current):
        # Choose the smallest possible current range.
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
        self.bias.command("SOURCE:CURRENT:AMPLITUDE " + str(current * 1e-3))
        self.bias_current = current

    def is_normal(self):
        if self.normal:
            return True
        else:
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
