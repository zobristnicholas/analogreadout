import numpy as np
from .JPL_Instruments import MG3692B_Signal_Generator as sig_gen
from .UCSB_Instruments import MSO6054A_Oscilloscope as scope


def do_iq_sweep(center, span, power, n_points):
    # initialize the signal generator and oscilloscope
    scope.initialize('iq_sweep')
    sig_gen.initalize(center, power)

    # generate list of frequencies to loop through
    freq_list = np.linspace(center - span / 2, center + span / 2, n_points)
    iq_list = np.zeros(freq_list.shape, dtype=np.complex)

    # loop through the frequencies and take data
    for index, freq in enumerate(freq_list):
        sig_gen.set_frequency(freq)
        iq_list[index] = scope.take_iq_point()

    return freq_list, iq_list
