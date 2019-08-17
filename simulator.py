import math
import os
import pickle
import random
import time

from model.Antenna import Antenna
from model.Configuration import Configuration
from model.Physics import Physics
from model.Target import Target


def generate_config():
    seed = random.randint(0, 1e6)
    physics = Physics(seed=seed, dt=0.01, total_time=60, c=3e8)
    global_clock_error_std = 1e-8
    target = Target(x0=0., y0=0., vx0=-10., vy0=0., pps=10, miss_rate=0.0, start_delay=0,
                    clock_error_std= global_clock_error_std)
    sigma_x = sigma_y = 1000.
    antennas = {i: Antenna(x=random.normalvariate(0, sigma_x), y=random.normalvariate(0, sigma_y),
                           clock_error_std=global_clock_error_std, delay=0, miss_rate=0.)
                for i in range(10)}
    lsq_std = (len(antennas) ** -0.5 * global_clock_error_std * physics.c / sigma_x) * 2 ** 0.5 * 1e4
    print "\t\t\t\tcov(x)=", lsq_std
    # antennas = {
    #     0: Antenna(x=-1000, y=1000, clock_error_std=1e-7, delay=10, miss_rate=0),
    #     1: Antenna(x=2000, y=0, clock_error_std=2e-6, delay=0, miss_rate=0),
    #     2: Antenna(x=3000, y=1000, clock_error_std=global_clock_error_std, delay=1e-9, miss_rate=0),
    #     3: Antenna(x=4000, y=10000, clock_error_std=global_clock_error_std, delay=100, miss_rate=0),
    #     4: Antenna(x=5000, y=0, clock_error_std=global_clock_error_std, delay=0, miss_rate=0),
    #     5: Antenna(x=6000, y=-10000, clock_error_std=global_clock_error_std, delay=3, miss_rate=0),
    #     6: Antenna(x=7000, y=-1000, clock_error_std=global_clock_error_std, delay=0.1, miss_rate=0),
    #     7: Antenna(x=8000, y=1000, clock_error_std=global_clock_error_std, delay=0.11, miss_rate=0),
    #     8: Antenna(x=10000, y=20000, clock_error_std=1e-8, delay=0.11, miss_rate=0),
    # }
    return Configuration(physics=physics, target=target, antennas=antennas, lsq_std=lsq_std)


def save_to_file(data, file_prefix, date):
    path = "results/{}".format(date)
    if not os.path.exists(path):
        os.makedirs(path)
    file_mame = "{file_prefix}.data".format(file_prefix=file_prefix)
    with open(path + "/" + file_mame, 'wb') as f:
        pickle.dump(data, f)


def calculate_real_coordinates(target, dt, measurements_counter):
    c = [[target.x0, target.vx0, 0, target.y0, target.vy0, 0, 0]]
    # c =[]
    for i in range(int(measurements_counter/3.)):
        cc = [c[-1][0] + dt * c[-1][1] + dt ** 2 / 2 * c[-1][2],
              c[-1][1] + dt * c[-1][2],
              # 0,
              10 * math.sin(2 * math.pi * dt * i / 60.),

              c[-1][3] + dt * c[-1][4] + dt ** 2 / 2 * c[-1][5],
              c[-1][4] + dt * c[-1][5],
              # -1 - 1e-3 * c[-1][4] * abs(c[-1][4]),
              10 * math.cos(2 * math.pi * dt * i / 60.),
              # 0,
              i * dt]
        # cc = [1000 * math.sin(i * dt/60),   1000 * math.cos(i * dt/60),   -1000 * math.sin(i * dt/60),
        #       -1000 * math.cos(i * dt/60),  1000 * math.sin(i * dt/60),   1000 * math.cos(i * dt/60),
        #       i*dt]
        c.append(cc)
    for i in range(int(measurements_counter/3.), int(2 * measurements_counter)):
        cc = [c[-1][0] + dt * c[-1][1] + dt ** 2 / 2 * c[-1][2],
              c[-1][1] + dt * c[-1][2],
              # 0,
              10 * math.sin(2 * math.pi * dt * i / 60.),

              c[-1][3] + dt * c[-1][4] + dt ** 2 / 2 * c[-1][5],
              c[-1][4] + dt * c[-1][5],
              # -1 - 1e-3 * c[-1][4] * abs(c[-1][4]),
              10 * math.cos(2 * math.pi * dt * i / 60.),
              # 0,
              i * dt]
        # cc = [1000 * math.sin(i * dt/60),   1000 * math.cos(i * dt/60),   -1000 * math.sin(i * dt/60),
        #       -1000 * math.cos(i * dt/60),  1000 * math.sin(i * dt/60),   1000 * math.cos(i * dt/60),
        #       i*dt]
        c.append(cc)
    return [(x, y, t) for (x, _, _, y, _, _, t) in c]


def calculate_measured_times(antennas, pulsing_coordinates, c):
    return {
        i_a: {
            i_p: t + ((x - antenna.x) ** 2 + (y - antenna.y) ** 2) ** 0.5 / c + antenna.delay + random.normalvariate(0,
                                                                                                                     antenna.clock_error_std)
            for (i_p, (x, y, t)) in pulsing_coordinates if random.random() > antenna.miss_rate
        }
        for (i_a, antenna) in antennas.items()
    }


def simulate_pulsing_coordinates(real_coordinates, target, dt, total_time):
    pulsing_coordinates = []
    i_p = 0
    for (_, _, t) in real_coordinates[int(round(target.start_delay / dt))::int(round(1. / target.pps / dt))]:
        current_pulse_time = t + random.normalvariate(0, target.clock_error_std)
        if current_pulse_time < total_time and random.random() > target.miss_rate:
            pulsing_coordinates.append((i_p, real_coordinates[int(round(current_pulse_time / dt))]))
            i_p += 1
    return pulsing_coordinates


def main():
    date = time.strftime('%Y%m%d-%H%M%S')
    print "generate_config..."
    config = generate_config()
    print "\tSAVING config..."
    save_to_file(config, 'configuration', date)
    print "\tDONE!"
    random.seed(config.physics.seed)
    measurements_counter = int(config.physics.total_time / config.physics.dt)
    print "calculate_real_coordinates [measurements_counter={}]...".format(measurements_counter)
    real_coordinates = calculate_real_coordinates(config.target, config.physics.dt, measurements_counter)
    print "\tSAVING real_coordinates..."
    save_to_file(real_coordinates, "real_coordinates", date)
    print "\tDONE!"
    print "simulate_pulsing_coordinates..."
    pulsing_coordinates = simulate_pulsing_coordinates(real_coordinates, config.target, config.physics.dt,
                                                       config.physics.total_time)
    print "\tSAVING pulsing_coordinates..."
    save_to_file(pulsing_coordinates, "pulsing_coordinates", date)
    print "\tDONE!"
    print "calculate_measured_times..."
    measured_times = calculate_measured_times(config.antennas, pulsing_coordinates, config.physics.c)
    print "\tSAVING measured_times..."
    save_to_file(measured_times, "measured_times", date)
    print "\tDONE!"
    print "date:", date
    print "seed=", config.physics.seed
    return date


if '__main__' == __name__:
    main()
