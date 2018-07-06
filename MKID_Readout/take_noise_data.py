# -*- coding: utf-8 -*-
from .JPL_Instruments import Signal_Generator_MG3692B as MG
from .JPL_Instruments import Programmable_Attenuator_83102042F as PA
from .UCSB_Instruments import Oscilloscope_MSO6054A as MSO


def take_noise_data(frequency, attenuation, offset, volts_per_div, n_triggers):
    scope_address = "GPIB0::7::INSTR"
    sig_gen_address = "GPIB0::4::INSTR"
    atten_address = "GPIB0::18::INSTR"
    
    # connect to signal generator, osilloscope, and attenuator
    scope = MSO(scope_address)
    sig_gen = MG(sig_gen_address)
    atten = PA(atten_address)

    # initialize the signal generator and oscilloscope
    scope.initialize('noise_data')
    atten.initialize(attenuation)
    sig_gen.initialize(frequency, 19)

    I_data, Q_data = scope.take_noise_data(offset, volts_per_div, n_triggers)

    return I_data, Q_data
