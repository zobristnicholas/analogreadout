def config(configuration):
    if configuration == "JPL":
        return jpl_config()
    elif configuration == "UCSB":
        return ucsb_config()
    else:
        raise ValueError("{} is not a recognized configuration type"
                         .format(configuration))


def jpl_config():
    scope_address = "GPIB0::7::INSTR"
    sig_gen_address = "GPIB0::4::INSTR"
    atten_address = "GPIB0::18::INSTR"

    dac_config = {"dac": {"instrument": "AnritsuMG3692B",
                          "arguments": (sig_gen_address),
                          "location": "signal_generators", "power": 14},
                  "attenuator": {"instrument": "Weinschel83102042F",
                                 "arguments": (atten_address, [1, 2]),
                                 "location": "attenuators"}}
    adc_config = {"adc": {"instrument": "AgilentMSO6054A", "arguments": (scope_address,),
                          "location": "oscilloscopes"}}
    sensor_config = {}  # thermometer, primary_amplifier
    procedure_config = {"sweep": "Sweep", "noise": "Noise", "pulse": "Pulse"}

    configuration = {"dac": dac_config, "adc": adc_config, "sensors": sensor_config,
                     "procedures": procedure_config}
    return configuration


def ucsb_config():
    sig_gen_arguments = [('USB0::0x0B5B::0xFFE0::084510::INSTR',),  # bottom (ch 1)
                         ('USB0::0x0B5B::0xFFE0::084511::INSTR',)]  # top (ch 2)
    sig_gen_types = ["AnritsuMG37022A", "AnritsuMG37022A"]
    atten_address = "GPIB2::10::INSTR"
    dac_config = {"dac": {"instrument": "MultipleSignalGenerators",
                          "arguments": (sig_gen_types, sig_gen_arguments),
                          "location": "signal_generators", "power": [14, 14]},
                  "attenuator": {"instrument": "Weinschel83102042F",
                                 "arguments": (atten_address, [1, 2]),
                                 "location": "attenuators"}}
    adc_config = {"adc": {"instrument": "NI6120", "arguments": (),
                          "location": "ni_digitizers"}}
    thermometer_address = 'visa://blackfridge.physics.ucsb.edu/ASRL1::INSTR'
    channel = 6
    scanner = '3716'
    sensor_config = {"thermometer": {"instrument": "LakeShore370AC",
                                     "arguments": (thermometer_address, channel, scanner),
                                      "location": "resistance_bridges"}}  
    procedure_config = {"sweep": "Sweep2", "noise": "Noise2", "pulse": "Pulse2"}
    configuration = {"dac": dac_config, "adc": adc_config, "sensors": sensor_config,
                     "procedures": procedure_config}
    return configuration
