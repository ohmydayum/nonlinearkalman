class Antenna(object):
    def __init__(self, x=0, y=0, clock_error_std=1, delay=0.1, miss_rate=0):
        self.miss_rate = miss_rate
        self.delay = delay
        self.clock_error_std = clock_error_std
        self.y = y
        self.x = x
