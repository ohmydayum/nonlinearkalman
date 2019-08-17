class Physics(object):
    def __init__(self, seed, dt=0.01, total_time=60., c=1):
        self.c = c
        self.total_time = total_time
        self.dt = dt
        self.seed = seed
