class LaserBox:
    OFF_STATE = (0, 0, 0, 0, 0)

    def __init__(self):
        pass

    def initialize(self, state):
        self.set_state(state)

    @staticmethod
    def set_state(state):
        if state is None:
            state = self.OFF_STATE
        pass

    def close(self):
        self.set_state(self.OFF_STATE)

    def reset(self):
        pass


class NotASource:
    def __init__(self, name=''):
        self.name = name
        message = "{} source is not connected and was not initialized"
        log.warning(message.format(self.name))

    def initialize(self, state):
        pass

    @staticmethod
    def set_state(state):
        pass

    def close(self):
        pass

    def reset(self):
        pass
