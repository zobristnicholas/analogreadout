import os
import warnings
import numpy as np
from datetime import datetime


def do_iq_sweep(daq, center, span, dac_atten, n_points, directory, adc_atten=0,
                power=None, verbose=True):
    save_string = '''\
{file_name}
center: {center}
span: {span}
dac_atten: {dac_atten}
adc_atten: {adc_atten}
power: {power}
n_points: {n_points}
'''

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    file_name = "sweep_" + timestamp + ".npz"
    file_path = os.path.join(directory, file_name)
    summary = save_string.format(**locals())
    metadata = locals()
    # save some data from the current state of the daq sensors
    save_state(metadata, daq)
    # get rid of some unnecessary information
    metadata.pop("daq")
    metadata.pop("save_string")

    # make center and span numpy arrays if needed
    if not isinstance(center, (list, tuple, np.ndarray)):
        center = np.array([center])
    if not isinstance(span, (list, tuple, np.ndarray)):
        span = np.array([span])
    if len(span) == 1 and len(center) != 1:
        span = span * np.ones(np.shape(center))
    if len(center) == 1 and len(span) != 1:
        center = center * np.ones(np.shape(span))

    # generate list of frequencies to loop through
    freq_list = np.zeros((len(center), n_points))
    for index, f0 in enumerate(center):
        freq_list[index, :] = np.linspace(f0 - span[index] / 2,
                                          f0 + span[index] / 2, n_points)
    freq_list = np.round(freq_list, 9)
    iq_list = np.zeros(freq_list.shape, dtype=np.complex)
    iq_offset = np.zeros(freq_list.shape, dtype=np.complex)

    # initialize the system in the right mode
    with warnings.catch_warnings():
        # ignoring warnings for setting infinite attenuation
        warnings.simplefilter("ignore", UserWarning)
        daq.initialize("sweep_data", freq_list[:, 0], power, np.inf, np.inf)
        

    # loop through the frequencies and take data
    for index, _ in enumerate(freq_list[0, :]):
        daq.dac.set_frequency(freq_list[:, index])
        iq_offset[:, index] = daq.adc.take_iq_point()
    
    # initialize the system in the right mode
    daq.initialize("sweep_data", freq_list[:, 0], power, dac_atten, adc_atten)
  
    # loop through the frequencies and take data
    for index, _ in enumerate(freq_list[0, :]):
        daq.dac.set_frequency(freq_list[:, index])
        iq_list[:, index] = daq.adc.take_iq_point()

    # save the data
    np.savez(file_path, freqs=freq_list, z=iq_list, z0=iq_offset, meta=metadata)
    if verbose:
        print(summary)
        
    return file_path


def take_noise_data(daq, frequency, dac_atten, n_triggers, directory, power=None,
                    adc_atten=0, sample_rate=2e6, verbose=True):

    save_string = '''\
{file_name}
frequency: {frequency}
dac_atten: {dac_atten}
adc_atten: {adc_atten}
power: {power}
n_triggers: {n_triggers}
sample_rate: {sample_rate}
'''

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    file_name = "noise_" + timestamp + ".npz"
    file_path = os.path.join(directory, file_name)
    summary = save_string.format(**locals())
    metadata = locals()
    # save some data from the current state of the daq sensors
    save_state(metadata, daq)
    # get rid of some unnecessary information
    metadata.pop("daq")
    metadata.pop("save_string")

    # initialize the system in the right mode
    daq.initialize("noise_data", frequency, power, dac_atten, adc_atten)

    # take the data
    I_data, Q_data = daq.adc.take_noise_data(n_triggers)

    # save the data
    np.savez(file_path, I_traces=I_data, Q_traces=Q_data, meta=metadata)

    if verbose:
        print(summary)
        
    return file_path


def take_pulse_data(daq, frequency, dac_atten, n_triggers, directory, power=None,
                    adc_atten=0, sample_rate=2e6, verbose=True):
    save_string = '''\
{file_name}
frequency: {frequency}
dac_atten: {dac_atten}
adc_atten: {adc_atten}
power: {power}
n_triggers: {n_triggers}
sample_rate: {sample_rate}
'''

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    file_name = "pulse_" + timestamp + ".npz"
    file_path = os.path.join(directory, file_name)
    summary = save_string.format(**locals())
    metadata = locals()
    # save some data from the current state of the daq sensors
    save_state(metadata, daq)
    # get rid of some unnecessary information
    metadata.pop("daq")
    metadata.pop("save_string")

    # initialize the system in the right mode
    daq.initialize("pulse_data", frequency, power, dac_atten, adc_atten)

    # take the data
    # TODO remove offset volts_per_div as required arguments
    I_data, Q_data = daq.adc.take_pulse_data(n_triggers)

    # save the data
    np.savez(file_path, I_traces=I_data, Q_traces=Q_data, meta=metadata)
    if verbose:
        print(summary)
        
    return file_path


def save_state(metadata, daq):
    metadata["thermometer"] = daq.thermometer.read_value()
    metadata["primary_amplifier"] = daq.primary_amplifier.read_value()
