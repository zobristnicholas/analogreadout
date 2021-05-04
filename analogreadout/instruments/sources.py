import socket
import logging

log = logging.getLogger(__name__)
log.addHandler(logging.NullHandler())


class LaserBox:
    """
    Arduino laser box controller.
    Notes:
        - The physical buttons on the box override the computer control when pressed.
        - The returned string is just the recieved data not the actual state.
        - The computer ip must be set to an ip similar to the Arduino.
          e.g. arduino: 10.200.130.7, computer: 10.200.130.X
    """
    OFF_STATE = (False, False, False, False, False)

    def __init__(self, ip='10.200.130.7', port=8888):
        self.address = (ip, port)
        self._connect()
        # self._clear_buffer()
        self.set_state(self.OFF_STATE)
        log.info("Connected to: Five Laser Box at %s port %s", ip, port)

    def initialize(self, state):
        if state is None:
            state = self.OFF_STATE
        self.set_state(state)

    def set_state(self, state):
        log.debug("Setting laser state to {}".format(state))
        # convert to bytes (first digit is the flipper that isn't used here)
        state = b'0' + b''.join(str(int(s)).encode() for s in state)
        assert len(state) == 6, "'state' argument must be a length 5 iterable of booleans"
        self.socket.sendto(state, self.address)
        try:
            returned_state, address = self.socket.recvfrom(self.buffer_size)
            assert state == returned_state
            assert address == self.address
        except (socket.timeout, AssertionError) as e:
            log.exception("Laser box state not confirmed: " + str(e))
            raise RuntimeError("Laser box state not confirmed")

    def close(self):
        try:
            self.set_state(self.OFF_STATE)
        except OSError:  # socket was already closed
            pass
        self.socket.close()

    def reset(self):
        self.close()
        self._connect()
        self.set_state(self.OFF_STATE)

    def _connect(self):
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.socket.settimeout(1)  # one second time out
        self.buffer_size = 6  # 6 bytes of information being returned


class NotASource:
    OFF_STATE = None
    
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
