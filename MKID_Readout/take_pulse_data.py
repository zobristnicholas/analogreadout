import os
import numpy as np
from datetime import datetime
from .JPL_Instruments import Signal_Generator_MG3692B as MG
from .JPL_Instruments import Programmable_Attenuator_83102042F as PA
from .UCSB_Instruments import Oscilloscope_MSO6054A as MSO

SAVE_STRING = '''\
{file_name}
frequency: {frequency}
attenuation: {attenuation}
power: {power}
offset: {offset}
volts_per_div: {volts_per_div}
n_triggers: {n_triggers}
trig_level: {trig_level}
slope: {slope}
trig_chan: {trig_chan}
sample_rate: {sample_rate}'''


def take_pulse_data(frequency, attenuation, power, offset, volts_per_div, n_triggers,
                    trig_level, slope, trig_chan, directory, sample_rate=2e6,
                    verbose=True):
    time = datetime.now()
    timestamp = time.strftime('%Y%m%d_%H%M%S')
    file_name = "pulse_" + timestamp + ".npz"
    file_path = os.path.join(directory, file_name)
    summary = SAVE_STRING.format(**locals())
    metadata = locals()

    scope_address = "GPIB0::7::INSTR"
    sig_gen_address = "GPIB0::4::INSTR"
    atten_address = "GPIB0::18::INSTR"
    
    # connect to signal generator, oscilloscope, and attenuator
    scope = MSO(scope_address)
    sig_gen = MG(sig_gen_address)
    atten = PA(atten_address)

    # initialize the signal generator and oscilloscope
    scope.initialize('pulse_data')
    atten.initialize(attenuation)
    sig_gen.initialize(frequency, power)

    # take the data
    I_data, Q_data = scope.take_pulse_data(offset, volts_per_div, n_triggers,
                                           trig_level, slope, trig_chan)
    # close the instruments
    scope.close()
    sig_gen.close()
    atten.close()

    # save the data
    np.savez(file_path, I_traces=I_data, Q_traces=Q_data, meta=metadata)
    if verbose:
        print(summary)