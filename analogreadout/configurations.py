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

    dac_config = {"dac": {"instrument": "AnritsuMG3692B", "arguments": (sig_gen_address,),
                          "location": "signal_generators", "power": 14},
                  "attenuator": {"instrument": "Weinschel83102042F",
                                 "arguments": (atten_address,),
                                 "location": "attenuators"}}
    adc_config = {"adc": {"instrument": "AgilentMSO6054A", "arguments": (scope_address,),
                          "location": "oscilloscopes"}}
    sensor_config = {}  # thermometer, primary_amplifier

    configuration = {"dac": dac_config, "adc": adc_config, "sensors": sensor_config}
    return configuration


def ucsb_config():
    dac_address = "GPIB0::7::INSTR"
    sig_gen_addresses = ["GPIB0::4::INSTR", "GPIB0::5::INSTR"]
    sig_gen_types = ["AnritsuMG3692B", "AnritsuMG3692B"]
    atten_address = "GPIB0::18::INSTR"
    dac_config = {"dac": {"instrument": "MultipleSignalGenerators",
                          "arguments": (sig_gen_addresses, sig_gen_types),
                          "location": "signal_generators", "power": 14},
                  "attenuator": {"type": "Weinschel83102042F",
                                 "arguments": (atten_address,)}}
    adc_config = {"adc": {"instrument": "AgilentMSO6054A", "arguments": (dac_address,),
                          "location": "adcs"}}
    sensor_config = {}  # thermometer, primary_amplifier

    configuration = {"dac": dac_config, "adc": adc_config, "sensors": sensor_config}
    raise NotImplementedError
