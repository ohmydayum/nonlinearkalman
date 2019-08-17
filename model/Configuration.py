from model import Physics, Target, Antenna


class Configuration(object):
    antennas = None  # type: Antenna
    target = None  # type: Target
    physics = None  # type: Physics

    def __init__(self, physics, target, antennas, lsq_std):
        self.lsq_std = lsq_std
        self.antennas = antennas
        self.target = target
        self.physics = physics
