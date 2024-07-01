#!/usr/bin/env python

import numpy as np
import pyhepmc
from argparse import ArgumentParser

import scipy.integrate
from src import io_handler, ph
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline, Akima1DInterpolator, CloughTocher2DInterpolator, SmoothBivariateSpline, RectBivariateSpline
from scipy.integrate import quad, simpson
from tqdm import tqdm
import scipy
from scipy.stats import rv_continuous

from matplotlib import cm


def sum_pdf_func(v, t, q_value, emin, single_func):
    # transformation of e1, e2 to eta1, eta2:
    # e1 = eta1+emin
    # e2 = eta2*(q_value-eta1-2*emin)+emin
    e2 = v*t/q_value
    e1 = t-e2
    eta1 = e1-emin
    eta2 = (e2-emin)/(q_value-eta1-2*emin)
    return 1/(q_value-eta1-2*emin) * t/q_value*single_func.__call__(eta1, eta2)


def rejection_sampler(pdf, x_min, x_max, pmax):
    while True:
        x = np.random.uniform(x_min, x_max)
        y = np.random.uniform(0, pmax)
        if y < pdf(x):
            return x


def weighted_sampler(pdf, x_min, x_max):
    x = np.random.uniform(x_min, x_max)
    p_unif = 1./(x_max - x_min)

    pdf_val = pdf(x)
    return (x, pdf_val/p_unif)


class spectrum_pdf_2d(rv_continuous):
    def __init__(self, eta1_grid: np.ndarray, eta2_grid: np.ndarray, data: np.ndarray):
        self.interp = CubicSpline(eta1_grid, eta2_grid, data)


class spectrum_pdf_1d(rv_continuous):
    def __init__(self, e_grid: np.ndarray, data: np.ndarray):
        self.e_grid = e_grid
        self.data = data
        if (len(e_grid) != len(data)):
            raise ValueError("e_grid and data must have the same length")
        self.interp = CubicSpline(e_grid, data)
        # compute cdf
        cdf_array = [0.0]
        e_grid_interp = [0.0]
        for i_e in range(1, len(e_grid)):
            cdf_dummy = simpson(
                y=self.data[:i_e], x=e_grid[:i_e])
            diff = cdf_dummy - cdf_array[-1]
            if (diff <= 0):
                continue
            cdf_array.append(cdf_dummy)
            e_grid_interp.append(e_grid[i_e])

        self.ppf_func = CubicSpline(
            np.array(cdf_array), np.array(e_grid_interp))
        self.cdf_func = CubicSpline(
            np.array(e_grid_interp), np.array(cdf_array))
        super().__init__(a=min(e_grid), b=max(e_grid))

    def _pdf(self, x):
        return self.interp(x)

    def _cdf(self, x):
        return self.cdf_func(x)

    def _ppf(self, q):
        return self.ppf_func(q)


def main():
    # read the spectrum data
    parser = ArgumentParser(
        description='Generate events from a spectrum file.')
    parser.add_argument('filename', type=str,
                        help='The filename to read', action='store')
    parser.add_argument('--ffs', default="Numeric", help='The Fermi functions to use',
                        choices=["Numeric", "PointLike", "ChargedSphere"])
    # argument for plotting spectra and cdf
    parser.add_argument('--plot', action='store_true',
                        help='Plot the spectra and cdf')

    # argument for number of events
    parser.add_argument('--n_events', type=int, default=1000,
                        help='The number of events to generate')

    args = parser.parse_args()

    data = io_handler.load_2d_spectra(args.filename)
    ffs = args.ffs
    if ffs not in data['Spectra']:
        print(f"Fermi function {ffs} not found in input file. Exiting.")
        return

    # Interpolate the spectra to create pdfs
    e1_grid = data['e1_grid']
    e2_grid = data['e2_grid']
    single_spectrum = np.array(
        data['Spectra'][ffs][ph.SPECTRUM_TYPES_NICE[ph.SINGLESPECTRUM]])
    angular_spectrum = np.array(
        data['Spectra'][ffs][ph.SPECTRUM_TYPES_NICE[ph.ANGULARSPECTRUM]])

    X = np.repeat(e1_grid, len(e2_grid))
    Y = np.tile(e2_grid, len(e1_grid))
    Z = np.reshape(single_spectrum, len(e1_grid)*len(e2_grid))

    # spl = SmoothBivariateSpline(X, Y, Z)
    spl_single = RectBivariateSpline(e1_grid, e2_grid, single_spectrum)
    spl_angular = RectBivariateSpline(e1_grid, e2_grid, angular_spectrum)
    if (args.plot):
        fig, ax = plt.subplots(figsize=(12, 6), ncols=2,
                               subplot_kw={"projection": "3d"})
        Z1 = spl_single.__call__(e1_grid, e2_grid)
        X, Y = np.meshgrid(e1_grid, e2_grid, indexing='ij')
        ax[0].plot_surface(X, Y, Z1, cmap=cm.coolwarm,
                           linewidth=0, antialiased=False)

        Z2 = spl_angular.__call__(e1_grid, e2_grid)
        ax[1].plot_surface(X, Y, Z2, cmap=cm.coolwarm,
                           linewidth=0, antialiased=False)
    # find coodinates and value of the maximum
    max_value = np.max(single_spectrum)
    max_eta1, max_eta_2 = np.unravel_index(
        np.argmax(single_spectrum, axis=None), single_spectrum.shape)
    e1_array = np.zeros(args.n_events)
    e2_array = np.zeros(args.n_events)
    t_array = np.zeros(args.n_events)
    cos_theta_array = np.zeros(args.n_events)
    weights_array = np.zeros(args.n_events)
    q_value = e1_grid[-1]+data["emin"]
    data_1d = io_handler.load_data("spectra.dat")
    angular_corr_factor = data_1d["PSFs"][ffs]["H"] / \
        data_1d["PSFs"][ffs]['G_single']
    rng = np.random.default_rng()
    for i_ev in tqdm(range(args.n_events)):
        # generate a random energy
        eta1 = rng.uniform(e1_grid[0], e1_grid[-1])
        eta2 = rng.uniform(0, 1)
        cos_theta_array[i_ev] = rng.uniform(-1, 1)
        e1_array[i_ev] = eta1+data['emin']
        e2_array[i_ev] = eta2*(q_value-eta1-2*data['emin'])+data['emin']
        w_single = spl_single.__call__(eta1, eta2)[0][0]
        w_angular = spl_angular.__call__(
            eta1, eta2)[0][0]*angular_corr_factor * cos_theta_array[i_ev]
        weights_array[i_ev] = w_single+w_angular
        t_array[i_ev] = e1_array[i_ev]+e2_array[i_ev]

    # load the single spectra
    data_1d = io_handler.load_data("spectra.dat")
    # e_grid = data_1d['energy_grid']
    # single_pdf = spectrum_pdf_1d(
    #     data_1d['energy_grid'], data_1d['Spectra'][ffs][ph.SPECTRUM_TYPES_NICE[ph.SINGLESPECTRUM]])
    # sum_pdf = spectrum_pdf_1d(
    #     data_1d['energy_grid'], data_1d['Spectra'][ffs][ph.SPECTRUM_TYPES_NICE[ph.SUMMEDSPECTRUM]])
    single_energy_spectrum = np.concatenate((e1_array, e2_array))

    e_grid = np.logspace(np.log10(min(single_energy_spectrum)),
                         np.log10(max(single_energy_spectrum)), 200)
    single_pdf = np.zeros_like(e_grid)
    sum_pdf = np.zeros_like(e_grid)
    for i_e in range(len(e_grid)):
        eta1_tmp = e_grid[i_e]-data['emin']
        single_pdf[i_e] = scipy.integrate.quad(lambda eta2, eta1: spl_single.__call__(eta1, eta2),
                                               0., 1.,
                                               args=eta1_tmp)[0]
        sum_pdf[i_e] = scipy.integrate.quad(lambda v: sum_pdf_func(
            v, e_grid[i_e], q_value, data['emin'], spl_single), 0, q_value)[0]

    fig, ax = plt.subplots(ncols=2, figsize=(10, 6))
    ax[0].hist(single_energy_spectrum, bins=100, weights=np.concatenate((weights_array, weights_array)),
               density=False,
               label='Single spectrum')
    ax[0].plot(e_grid, single_pdf*args.n_events *
               0.01*2, label='Single pdf')

    ax[1].hist(t_array, bins=100, density=True,
               weights=weights_array, label='Sum spectrum')
    ax[1].plot(e_grid, sum_pdf, label='Sum pdf')

    fig, ax = plt.subplots(ncols=2, figsize=(10, 6))
    ax[0].hist(cos_theta_array, bins=100, weights=weights_array,
               density=True,
               label='Cos theta')
    ctheta = np.linspace(-1, 1, 100)
    print(angular_corr_factor)
    ax[0].plot(ctheta, 0.5+0.5*angular_corr_factor *
               ctheta, label='Uniform pdf')

    # compute K from simulations
    wPlus = np.sum(weights_array[cos_theta_array <= 0])
    wMinus = np.sum(weights_array[cos_theta_array > 0])
    wTotal = wMinus+wPlus
    print(f"{wMinus:.2f} {wPlus:.2f}")
    print(f"K = {-2*(wPlus-wMinus)/wTotal:.2f}")

    # plot distribution of weights
    fig, ax = plt.subplots(figsize=(10, 6))
    print(np.median(weights_array))
    print(np.mean(weights_array))
    print(np.std(weights_array))
    ax.hist(np.log10(weights_array), bins=100, density=False, label='Weights')
    plt.show()


if __name__ == '__main__':
    main()
    plt.show()
