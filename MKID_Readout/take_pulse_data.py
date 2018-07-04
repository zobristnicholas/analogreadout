from .JPL_Instruments import MG3692B_Signal_Generator as sig_gen
from .UCSB_Instruments import MSO6054A_Oscilloscope as scope


def take_pulse_data(frequency, power):
    # initialize the signal generator and oscilloscope
    scope.initialize('iq_sweep')
    sig_gen.initalize(frequency, power)

    data = scope.take_pulse_data()

    return data
