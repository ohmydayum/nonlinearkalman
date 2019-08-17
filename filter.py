import pickle
import random
import time

import numpy
from bokeh import plotting
from bokeh.models import HoverTool
from bokeh.palettes import Category10 as PALETTES
from scipy.linalg import cholesky
from scipy.optimize import least_squares
from scipy.sparse import block_diag, vstack, eye, hstack, coo_matrix, lil_matrix, diags, identity, csr_matrix
from scipy.sparse.linalg import lsqr

import simulator
from model.Configuration import Configuration


def read_data(file, date):
    filename = "results/{date}/{file}.data".format(date=date, file=file)
    with open(filename, "rb") as f:
        return pickle.load(f)


def f(p0, p):
    return numpy.linalg.norm(p0 - p)


def f_dx(p0, p):
    return float(p[0] - p0[0]) / f(p0, p)


def f_dy(p0, p):
    return float(p[1] - p0[1]) / f(p0, p)


def calculate_lsq(all_measurements, config, t0):
    lsq = []
    errors = []
    guess = numpy.array([[0], [0], [t0]], dtype='float')
    C = config.physics.c ** 2 * numpy.diag(
        [antenna.clock_error_std ** 2 for i_a, antenna in sorted(config.antennas.items())])
    C_inv = numpy.linalg.inv(C)
    j = 0
    for measurements in all_measurements:
        A = numpy.ones((0, 3), dtype='float')
        b = numpy.ones((0, 1), dtype='float')
        for i_a, antenna in filter(lambda (i_a, a): measurements[i_a] is not None, sorted(config.antennas.items())):
            p0 = numpy.array([[antenna.x], [antenna.y]])
            # print numpy.abs(f(p0, p)-config.physics.c*(all_measurements[j-1][i_a]))
            p = guess[0:2]
            A_current = numpy.array([[f_dx(p0=p0, p=p), f_dy(p0=p0, p=p), config.physics.c]], dtype='float')
            A = numpy.concatenate([A, A_current])
            b_current = numpy.array([config.physics.c * (measurements[i_a] - guess[2] - antenna.delay) - f(p0=p0, p=p)],
                                    dtype='float')
            b = numpy.concatenate([b, b_current])
        try:
            #  TODO
            # current_coordinate = lsq[-1] + least_squares(residuals, lsq[-1], args=(measurements, config.antennas))
            M = numpy.linalg.inv(A.T.dot(C_inv).dot(A)).dot(A.T).dot(C_inv)
            current_coordinate = guess + M.dot(b)
            guess = current_coordinate
            lsq.append(current_coordinate)
            current_error = M.dot(C).dot(M.T)
            errors.append(current_error)
        except Exception as e:
            print "lsq error:", e
        j += 1
    return lsq, errors


def plot_measurements(figure, colors_palette, config, measured_times, pulses_times, counter=None):
    for i_a, antenna in sorted(config.antennas.items()):
        for i_m, current_time_measurement in sorted(measured_times[i_a].items()):
            i_m = int(i_m)
            if i_m is not None and not i_m < counter:
                continue
            current_pulse_time = pulses_times[i_m] + antenna.delay
            inner_radius = max(0,
                               current_time_measurement - 3 * antenna.clock_error_std - current_pulse_time) * config.physics.c
            outer_radius = max(0,
                               current_time_measurement + 3 * antenna.clock_error_std - current_pulse_time) * config.physics.c
            try:
                figure.annulus(antenna.x, antenna.y,
                               color=colors_palette[3], legend="antennas measurements",
                               outer_radius=outer_radius, inner_radius=inner_radius, fill_alpha=0.2)
            except Exception as e:
                print "plot error:", e


def plot_antennas(figure, colors_palette, config):
    for i_a, antenna in config.antennas.items():
        figure.circle(antenna.x, antenna.y, color=colors_palette[3],
                      legend="antennas", line_width=15, line_alpha=0.9)


def plot_target(figure, colors_palette, real_coordinates, pulsing_coordinates):
    figure.line([x for (x, y, t) in real_coordinates],
                [y for (x, y, t) in real_coordinates],
                color=colors_palette[1],
                legend="real", line_width=5, line_alpha=0.4)
    figure.circle_cross([x for (i_p, (x, y, t)) in pulsing_coordinates],
                        [y for (i_p, (x, y, t)) in pulsing_coordinates],
                        color=colors_palette[2],
                        legend="pulsing_coordinates", line_width=5, line_alpha=0.5)


def calculate_kalman_filter(measurements, measurements_error):
    # measure_error_std = 300.0 #config.physics.c * (config.target.clock_error_std+config.antennas[0].clock_error_std)  # TODO: who knows, really
    # x = [numpy.array(
    #                 [measurements[0][0], (measurements[1][0]-measurements[0][0])/(measurements[1][2]-measurements[0][2]), (measurements[1][0]-measurements[0][0])/(measurements[1][2]-measurements[0][2]),
    #                  measurements[0][1], (measurements[1][1] - measurements[0][1]) / (measurements[1][2] - measurements[0][2]), (measurements[1][0]-measurements[0][0])/(measurements[1][2]-measurements[0][2]),]
    #                )]
    x = [numpy.array([measurements[0][0], [0], [0], measurements[0][1], [0], [0]], dtype='float')]
    P = (measurements_error[0][0] + measurements_error[0][1]) ** 0.5 * numpy.identity(6)  # TODO
    p_mvkf = [P]
    H = numpy.array([[1., 0., 0, 0, 0., 0],
                     [0., 0., 0, 1, 0., 0]], dtype='float')
    # TODO: this is just working random Q matrix :(
    Q = numpy.diag([0., 0, 1, 0, 0, 1])
    for i in range(1, len(measurements)):
        t_delta = measurements[i][2][0] - measurements[i - 1][2][0]
        F = numpy.array([[1., t_delta, t_delta ** 2 / 2, 0, 0, 0],
                         [0, 1, t_delta, 0, 0, 0],
                         [0, 0, 1, 0, 0, 0],
                         [0, 0, 0, 1, t_delta, t_delta ** 2 / 2],
                         [0, 0, 0, 0, 1, t_delta],
                         [0, 0, 0, 0, 0, 1]])
        x_prior = F.dot(x[i - 1])
        P = F.dot(P).dot(F.T) + Q

        R = numpy.diag([measurements_error[i][0], measurements_error[i][1]])
        S = H.dot(P).dot(H.T) + R
        K = P.dot(H.T).dot(numpy.linalg.inv(S))
        res = measurements[i][:2] - H.dot(x_prior)
        x_posterior = x_prior + K.dot(res)
        P = P - K.dot(H).dot(P)

        x.append(x_posterior)
        p_mvkf.append(P)
    return x, p_mvkf


def calculate_mean_absolute_error(real, param):
    return float(numpy.mean([numpy.linalg.norm([r[0] - p[0], r[1] - p[1]]) for (r, p) in zip(real, param)]))


def calculate_kalman_smoother(t0, measurements, measurements_errors):
    k = len(measurements)
    b = numpy.array([reduce(lambda so_far, o: so_far + o, [[x, y] + 6 * [0] for (x, y, t) in measurements])],
                    dtype='float').T
    b = b[:-6]
    Q = numpy.diag([1, 1, 1, 1, 1, 1]) * 1e-1
    H = numpy.array([[1., 0., 0, 0, 0., 0],
                     [0., 0., 0, 1, 0., 0]], dtype='float')
    Z = numpy.zeros((2, 6))
    t_delta = measurements[0][2][0] - t0
    WHs = []
    WFs = []
    WIs = []
    WZs = []
    Ws = [(cholesky(numpy.linalg.inv(m[:2, :2])), cholesky(numpy.linalg.inv(Q))) for m in measurements_errors]
    for i in range(1, k):
        F = numpy.array([[1., t_delta, t_delta ** 2 / 2., 0, 0, 0],
                         [0, 1., t_delta, 0, 0, 0],
                         [0, 0, 1., 0, 0, 0],
                         [0, 0, 0, 1., t_delta, t_delta ** 2 / 2.],
                         [0, 0, 0, 0, 1., t_delta],
                         [0, 0, 0, 0, 0, 1.]])
        t_delta = measurements[i][2][0] - measurements[i - 1][2][0]

        WFs.append(Ws[i][1].dot(F))
        WIs.append(-Ws[i][1])
        WZs.append(Z)
        WHs.append(Ws[i][0].dot(H))
    WA1 = block_diag(mats=[vstack([f, z]) for (f, z) in zip(WFs, WZs)])
    Z_COLUMN = coo_matrix(numpy.zeros((WA1.shape[0], 6)))
    WA1 = hstack([WA1, Z_COLUMN])
    WA2 = block_diag(mats=[vstack([i, h]) for (i, h) in zip(WIs, WHs)])
    WA2 = hstack([WA2, Z_COLUMN])
    offset_identity = eye(WA2.shape[1], k=6)
    WA2 = WA2.dot(offset_identity)
    WA = WA1 + WA2
    WH0 = Ws[0][0].dot(H)
    Z0 = coo_matrix(numpy.zeros((2, WA.shape[1] - 6)))
    WH0_Z0 = hstack([WH0, Z0])
    WA = vstack([WH0_Z0, WA])
    W = block_diag(
        mats=[block_diag(mats=[cholesky(numpy.linalg.inv(m[:2, :2])), cholesky(numpy.linalg.inv(Q))]) for m in
              measurements_errors[:-1]] + [cholesky(numpy.linalg.inv(measurements_errors[-1][:2, :2]))])
    Wb = W.dot(b)
    ks = lsqr(WA, Wb)[0]
    return ks.reshape(ks.size / 6, 6)


# TODO
def calculate_non_linear_solver(p0, measurements, antennas, dt, c):
    def calculate_kalman_residuals(p, state_size, counter_antennas, antennas, counter_pulses, measurements, dt, c):
        r = 1./c
        residuals = []
        for i in range(counter_pulses - 1):  # foreach pulse except the last
            [x, vx, ax, y, vy, ay, t] = p[state_size * i: state_size * (i + 1)]
            [xn, vxn, axn, yn, vyn, ayn, tn] = p[state_size * (i + 1): state_size * (i + 2)]
            for j in range(counter_antennas):  # foreach antenna
                if measurements[i][j] is None:
                    residuals.append([0])
                else:
                    residuals.append([r * antennas[j].clock_error_std ** -0.5 *
                                      (measurements[i][j] * c - t * c -
                                       numpy.linalg.norm([antennas[j].x - x, antennas[j].y - y]))])
            t_diff = (tn - t)
            residuals.append([xn - (x + t_diff * vx + t_diff ** 2 / 2 * ax)])
            residuals.append([vxn - (vx + t_diff * ax)])
            residuals.append([axn - (ax)])
            # residuals.append([0])
            # residuals.append([0])
            residuals.append([yn - (y + t_diff * vy + t_diff ** 2 / 2 * ay)])
            residuals.append([vyn - (vy + t_diff * ay)])
            residuals.append([ayn - (ay)])
            # residuals.append([0])
            # residuals.append([0])
            # residuals.append([0])
            # residuals.append([0])
            residuals.append([tn  - (t + dt)])

        [x, _, _, y, _, _, t] = p[-state_size:]
        for j in range(counter_antennas):  # foreach antenna
            if measurements[counter_pulses - 1][j] is None:
                residuals.append([0])
            else:
                residuals.append([r * antennas[j].clock_error_std ** -0.5 *
                                  (measurements[counter_pulses - 1][j] * c - t * c -
                                   numpy.linalg.norm([antennas[j].x - x, antennas[j].y - y]))])
        return numpy.array(residuals).reshape(len(residuals))

    def calculate_jacobian_sparsity_matrix(counter_antennas, counter_pulses, state_size):
        k1 = (counter_antennas + state_size) * counter_pulses - state_size
        k2 = state_size * counter_pulses
        sparsity = lil_matrix((k1, k2), dtype=int)
        row_start = column_start = 0
        for i in range(counter_pulses):
            sparsity[row_start:row_start + counter_antennas,
                    [column_start, column_start + state_size / 2, column_start + state_size - 1]] = 1

            row_start += counter_antennas

            sparsity[row_start:row_start + state_size / 2,
                    column_start:column_start + state_size / 2] = 1

            sparsity[row_start:row_start + state_size / 2,
                    column_start + state_size:column_start + state_size + state_size / 2] = 1 #identity(state_size/2)

            sparsity[row_start:row_start + state_size / 2 - 1,
                    [column_start + state_size - 1, column_start + 2 * state_size - 1]] = 1

            row_start += state_size / 2
            column_start += state_size / 2

            sparsity[row_start:row_start + state_size / 2,
                    column_start:column_start + state_size / 2] = 1

            sparsity[row_start:row_start + state_size / 2,
                    column_start + state_size:column_start + state_size + state_size / 2] = 1 # identity(state_size / 2)

            sparsity[row_start:row_start + state_size / 2 - 1,
            [column_start + state_size/ 2 + 1, column_start + state_size / 2 + state_size ]] = 1

            # sparsity[row_start + state_size / 2 - 2,
            # [column_start + state_size / 2 , column_start + state_size / 2 + state_size]] = 1

            row_start += state_size / 2 + 1
            column_start += state_size / 2 + 1
        return sparsity

    state_size = 7
    counter_antennas = len(antennas)
    counter_pulses = len(measurements)
    solution = least_squares(fun=calculate_kalman_residuals,
                             jac_sparsity=calculate_jacobian_sparsity_matrix(counter_antennas, counter_pulses,
                                                                             state_size),
                             # method='lm',
                             tr_solver='lsmr',
                             xtol=1e-30,
                             ftol=1e-30,
                             gtol=1e-30,
                             # loss = 'soft_l1',
                             # loss = 'cauchy',
                             args=(state_size, counter_antennas, antennas, counter_pulses, measurements, dt, c),
                             x0=p0, verbose=2)
    # print solution
    p = solution.x
    return p.reshape(counter_pulses, state_size)


if '__main__' == __name__:
    date = simulator.main()
    print date
    config = read_data('configuration', date)  # type: Configuration
    _real_coordinates = read_data('real_coordinates', date)
    _pulsing_coordinates = read_data('pulsing_coordinates', date)
    measured_times = read_data('measured_times', date)

    hover = HoverTool(tooltips=[
        ("index", "$index"),
        ("(x,y)", "(@x, @y)"),
    ])
    figure = plotting.figure(
        tools=["pan,box_zoom,zoom_out,reset,save,crosshair",
               hover,
               ],
        title="Reverse GPS simulation- {}".format(date), width=900, height=900,
        x_axis_label='x [m]', y_axis_label='y [m]',
        x_range=(-4e3, 6e3), y_range=(-4e3, 2e3)
    )
    colors_palette = PALETTES[10]
    plot_antennas(figure, colors_palette, config)
    plot_target(figure, colors_palette, _real_coordinates, _pulsing_coordinates)
    # plot_measurements(figure, colors_palette, config, measured_times, [t for (_, (_, _, t)) in _pulsing_coordinates],
    #                   counter=1)
    pulses_indices = list(set(reduce(lambda x, y: x + y,
                                     [[i_p for (i_p, _) in antenna_measurements.items()]
                                      for (i_a, antenna_measurements) in measured_times.items()])))
    all_measurements = numpy.asarray([
        [
            measured_times[i_a][i_m] if i_m in measured_times[i_a] else None
            for i_a in range(len(config.antennas))
        ] for i_m in pulses_indices])
    print "calculate_lsq..."
    lsq, lsq_errors = calculate_lsq(all_measurements, config, config.target.start_delay)
    lsq_theory_x_std = numpy.mean([m[0][0] ** 0.5 for m in lsq_errors])
    print "\t\t\t\tlsq_theory_x_std=", lsq_theory_x_std
    lsq_theory_y_std = numpy.mean([m[1][1] ** 0.5 for m in lsq_errors])
    # lsq_theory_std = round(lsq_theory_x_std + lsq_theory_x_std, 2)
    lsq_times_in_dt = [int(round(t / config.physics.dt)) for (x, y, t) in lsq]
    real_positions_at_lsq_times = [_real_coordinates[t] for t in lsq_times_in_dt]
    lsq_mae = round(calculate_mean_absolute_error(real_positions_at_lsq_times, [(x, y) for ([x], [y], t) in lsq]), 2)
    lsq_legend = "LS [mae={}, theory_std={}]".format(lsq_mae, config.lsq_std)
    # figure.circle([p[0][0] for p in lsq], [p[1][0] for p in lsq], fill_color=colors_palette[8],
    #               legend="LS error", size=[(m[0][0] + m[1][1]) ** 0.5 for m in lsq_errors], fill_alpha=0.2,
    #               line_alpha=0)
    figure.line([p[0][0] for p in lsq], [p[1][0] for p in lsq], color=colors_palette[8],
                legend=lsq_legend, line_width=5, line_alpha=0.5)
    figure.circle([p[0][0] for p in lsq], [p[1][0] for p in lsq], color=colors_palette[8],
                  legend=lsq_legend, line_width=5, line_alpha=0.5)
    print "\t", lsq_legend

    print "calculate_kalman_filter"
    kf, p_kf = calculate_kalman_filter(lsq, [(m[0][0], m[1][1]) for m in lsq_errors])
    kf_mae = calculate_mean_absolute_error(real_positions_at_lsq_times, [(x, y) for (x, _, _, y, _, _) in kf])
    kf_legend = "LS + Multivariate Kalman Filter [mae={}]".format(round(kf_mae), 2)
    # figure.circle([k[0][0] for k in kf], [k[3][0] for k in kf], fill_color=colors_palette[10],
    #               legend="LS + Multivariate Kalman Filter error", size=[p[0][0] ** 0.5 for p in p_kf], fill_alpha=0.2,
    #               line_alpha=0)
    kf_l = figure.line([x for ([x], _, _, _, _, _) in kf], [y for (_, _, _, [y], _, _) in kf], color=colors_palette[9],
                       legend=kf_legend, line_width=5)
    kf_c = figure.circle([x for ([x], _, _, _, _, _) in kf], [y for (_, _, _, [y], _, _) in kf],
                         color=colors_palette[9],
                         legend=kf_legend, line_width=5)
    print "\t", kf_legend
    kf_l.muted = True
    kf_c.muted = True

    print "calculate_kalman_smoother"
    # ks = calculate_kalman_smoother_PS(lsq, lsq_errors)
    ks = calculate_kalman_smoother(config.target.start_delay,
                                   [(x, y, t) for (([x], _, _, [y], _, _), (_, _, t)) in zip(kf, lsq)],
                                   lsq_errors)
    ks_mae = calculate_mean_absolute_error(real_positions_at_lsq_times, [(x, y) for (x, _, _, y, _, _) in ks])
    ks_legend = "LS + Kalman filter + Multivariate Kalman Smoothing [mae={}]".format(round(ks_mae), 2)
    figure.line([x for (x, _, _, _, _, _) in ks], [y for (_, _, _, y, _, _) in ks], color=colors_palette[3],
                legend=ks_legend, line_width=5, line_alpha=0.4)
    figure.circle([x for (x, _, _, _, _, _) in ks], [y for (_, _, _, y, _, _) in ks], color=colors_palette[3],
                  legend=ks_legend, line_width=5, line_alpha=0.5)
    print "\t", ks_legend

    ks2 = calculate_kalman_smoother(config.target.start_delay, lsq, lsq_errors)
    ks_mae2 = calculate_mean_absolute_error(real_positions_at_lsq_times, [(x, y) for (x, _, _, y, _, _) in ks2])
    ks_legend2 = "LS + Multivariate Kalman Smoothing [mae={}]".format(round(ks_mae2), 2)
    figure.line([x for (x, _, _, _, _, _) in ks2], [y for (_, _, _, y, _, _) in ks2], color=colors_palette[4],
                legend=ks_legend2, line_width=5, line_alpha=0.4)
    figure.circle([x for (x, _, _, _, _, _) in ks2], [y for (_, _, _, y, _, _) in ks2], color=colors_palette[4],
                  legend=ks_legend2, line_width=5, line_alpha=0.5)
    print "\t", ks_legend2

    print "calculate_non_linear_solver"
    # p0 = [[x, vx, ax, y, vy, ay, t] for (([x], [vx], [ax], [y], [vy], [ay]), (_, _, t)) in zip(kf, lsq)]
    # p0 = numpy.array(kf).reshape(len(kf)*len(kf[0]))
    # p0 = numpy.array(ks).reshape(len(ks)*len(ks[0]))
    p0 = numpy.array(reduce(lambda a, b: a + b, [[x, 0, 0, y, 0, 0, t] for ([x], [y], [t]) in lsq]))
    # p0 = [list(ks2[i]) + [lsq[i][2]] for i in range(len(ks2))]
    # p0 = numpy.array(p0).reshape(len(p0) * len(p0[0]))
    # print "#0"
    nls = calculate_non_linear_solver(p0, all_measurements, config.antennas, 1.0 / config.target.pps,
                                      config.physics.c)
    nls_times_in_dt = [int(round(t / config.physics.dt)) for [_,_,_,_,_,_,t] in nls]
    real_positions_at_nls_times = [_real_coordinates[t] for t in nls_times_in_dt]

    nls_mae = calculate_mean_absolute_error(real_positions_at_nls_times, [(x, y) for (x, _, _, y, _, _, _) in nls])
    # for i in range(1, 3):
    #     print "#",i
    #     p0 = [s*random.normalvariate(1, 0.1) for s in p]
    #     t_nls = calculate_non_linear_solver(p0, all_measurements, config.antennas, 1.0 / config.target.pps, config.physics.c)
    #     t_nls_mae = calculate_mean_absolute_error(real_positions_at_lsq_times, [(x, y) for (x, _, _, y, _, _) in t_nls])
    #     if t_nls_mae < nls_mae:
    #         nls_mae = t_nls_mae
    #         nls = t_nls
    nls_legend = "Non-linear state-space markov solver [mae={}]".format(round(nls_mae), 2)
    figure.line([x for (x, _, _, y, _, _, _) in nls], [y for (_, _, _, y, _, _, _) in nls], color=colors_palette[7],
                legend=nls_legend, line_width=5, line_alpha=0.4)
    figure.circle([x for (x, _, _, y, _, _, _) in nls], [y for (_, _, _, y, _, _, _) in nls], color=colors_palette[7],
                  legend=nls_legend, line_width=5, line_alpha=0.5)
    print "\t", nls_legend

    figure.legend.location = "top_right"
    figure.legend.click_policy = "hide"
    print "showing..."
    plotting.output_file('results/{}/graph_{}.html'.format(date, time.strftime('%Y%m%d-%H%M%S')))  # , mode='inline')
    plotting.show(figure)

    # print calculate_advanced_kalman_smoother([6,0,0,6,0,0,9,0,0,9,0,0], [[5,5], [10,0]], [Antenna(x=0, y=0), Antenna(x=6, y=8)], 0.1, 1)
