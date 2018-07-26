import os
import numpy as np
from time import sleep
from datetime import datetime
from .JPL_Instruments import Signal_Generator_MG3692B as MG
from .JPL_Instruments import Programmable_Attenuator_83102042F as PA
from .UCSB_Instruments import Oscilloscope_MSO6054A as MSO

SAVE_STRING = '''\
{file_name}
center: {center}
span: {span}
attenuation: {attenuation}
power: {power}
n_points: {n_points}'''


def do_iq_sweep(center, span, attenuation, power, n_points, directory, verbose=True):
    time = datetime.now()
    timestamp = time.strftime('%Y%m%d_%H%M%S')
    file_name = "iq_sweep_" + timestamp + ".npz"
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
    atten.reset()
    scope.initialize('iq_sweep')
    sig_gen.initialize(center, power)

    # generate list of frequencies to loop through
    freq_list = np.linspace(center - span / 2, center + span / 2, n_points)
    freq_list = np.round(freq_list, 6)
    df = np.round(np.mean(np.diff(freq_list)), 7)

    iq_list = np.zeros(freq_list.shape, dtype=np.complex)
    iq_offset = np.zeros(freq_list.shape, dtype=np.complex)

    # set increment
    sig_gen.set_increment(df)
    
    # take initial point a few times to get the scope in the right scale range
    sig_gen.set_frequency(freq_list[0])
    for _ in range(30):
        scope.take_iq_point()
   

    # loop through the frequencies and take data
    for index, frequency in enumerate(freq_list):
        if index != 0:
            sig_gen.increment()
        scope.take_iq_point()
        iq_offset[index] = scope.take_iq_point()
    
    # initialize the attenuator
    atten.initialize(attenuation)
    
    # take initial point a few times to get the scope in the right scale range
    sig_gen.set_frequency(freq_list[0])
    for _ in range(30):
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

    # save the data
    np.savez(file_path, freqs=freq_list, z=iq_list, z0=iq_offset, meta=metadata)
    if verbose:
        print(summary)
    return file_path
