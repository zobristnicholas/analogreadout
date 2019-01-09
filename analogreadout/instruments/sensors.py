import logging
import numpy as np

log = logging.getLogger(__name__)
log.addHandler(logging.NullHandler())

class NotASensor:
    def __init__(self, name=''):
        self.name = name
        message = "{} sensor is not connected and was not initialized"
        log.warning(message.format(self.name))

    def initialize(self):
        pass

    @staticmethod
    def read_value():
        return np.nan

    def close(self):
        pass

    def reset(self):
        pass
