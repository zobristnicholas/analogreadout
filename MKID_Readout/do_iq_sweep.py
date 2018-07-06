import numpy as np
from time import sleep
from .JPL_Instruments import Signal_Generator_MG3692B as MG
from .JPL_Instruments import Programmable_Attenuator_83102042F as PA
from .UCSB_Instruments import Oscilloscope_MSO6054A as MSO


def do_iq_sweep(center, span, attenuation, n_points):
    scope_address = "GPIB0::7::INSTR"
    sig_gen_address = "GPIB0::4::INSTR"
    atten_address = "GPIB0::18::INSTR"
    
    # connect to signal generator, osilloscope, and attenuator
    scope = MSO(scope_address)
    sig_gen = MG(sig_gen_address)
    atten = PA(atten_address)
    
    # initialize the signal generator and oscilloscope
    scope.initialize('iq_sweep')
    sig_gen.initialize(center, 19)
    atten.write("*RST")

    # generate list of frequencies to loop through
    df = round(span / n_points, 6)
    freq_list = np.arange(center - span / 2, center + span / 2, df)
    iq_list = np.zeros(freq_list.shape, dtype=np.complex)
    iq_offset = np.zeros(freq_list.shape, dtype=np.complex)


    # set increment
    sig_gen.set_increment(df)
    
    # take initial point a few times to get the scope in the right scale range
    sig_gen.set_frequency(freq_list[0])
    scope.take_iq_point()
    scope.take_iq_point()
    scope.take_iq_point()
    
    # loop through the frequencies and take data
    for index, frequency in enumerate(freq_list):
        if index != 0:
            sig_gen.increment()
        sleep(0.02)
        scope.take_iq_point()
        iq_offset[index] = scope.take_iq_point()
    
    #initialize the attenuator
    atten.initialize(attenuation)
    sleep(1)
    
    # take initial point a few times to get the scope in the right scale range
    sig_gen.set_frequency(freq_list[0])
    scope.take_iq_point()
    scope.take_iq_point()
    scope.take_iq_point()
    
    # loop through the frequencies and take data
    for index, frequency in enumerate(freq_list):
        if index != 0:
            sig_gen.increment()
        sleep(0.02)
        scope.take_iq_point()
        iq_list[index] = scope.take_iq_point()
    scope.close()
    sig_gen.close()
    atten.close()

    return freq_list, iq_list, iq_offset
