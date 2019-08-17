class Target(object):
    def __init__(self, x0=0., y0=0., vx0=0., vy0=0., pps=1., miss_rate=0, start_delay=0, clock_error_std=0):
        self.miss_rate = miss_rate
        self.start_delay = start_delay
        self.clock_error_std = clock_error_std
        self.pps = pps
        self.vy0 = vy0
        self.vx0 = vx0
        self.y0 = y0
        self.x0 = x0
