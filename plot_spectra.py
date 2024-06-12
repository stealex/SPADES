#!/usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from argparse import ArgumentParser
from src import ph

# Load the data from the file


def load_data(filename):
    with open(filename, 'r') as f:
        lines = f.readlines()

    data = {}
    data['PSFs'] = {}
    data['Spectra'] = {}

    headers_psfs = []
    headers_spectra = []
    read_psfs = False
    read_spectra = False
    i_point = 0

    for line in lines:
        if line[0] == '#':
            if "PSFs:" in line:
                data['PSFs'] = {}
                read_psfs = True
                read_spectra = False
                continue
            elif "Spectra:" in line:
                data['Spectra'] = {}
                read_psfs = False
                read_spectra = True
                continue
            elif 'N energy points' in line:
                tokens = line.strip().split("=")
                data["energy_grid"] = np.zeros(int(tokens[1]))
                continue
            else:
                continue

        if read_psfs:
            tokens = line.strip().split()
            if tokens[0][0] == 'G':
                # means we're reading header line
                for tok in tokens:
                    headers_psfs.append(tok)
                    # split at brackets
                    toks = tok.split('(')
                    key_ff = toks[1].strip(')')
                    sp_type = toks[0].strip()
                    if key_ff in data['PSFs']:
                        data['PSFs'][key_ff][sp_type] = 0.
                    else:
                        data['PSFs'][key_ff] = {sp_type: 0.}
            else:
                for i, tok in enumerate(tokens):
                    toks = headers_psfs[i].split('(')
                    key_ff = toks[1].strip(')')
                    sp_type = toks[0].strip()

                    data['PSFs'][key_ff][sp_type] = float(tok)
                read_psfs = False

        if read_spectra:
            tokens = line.strip().split()
            if tokens[0][0] == 'E':
                # prepare arrays
                for i, tok in enumerate(tokens):
                    if i == 0:
                        continue
                    headers_spectra.append(tok)
                    toks = tok.split('(')
                    key_ff = toks[1].strip(')')
                    sp_type = toks[0].strip()
                    if not (key_ff in data['Spectra']):
                        data['Spectra'][key_ff] = {sp_type: np.zeros_like(
                            data["energy_grid"])}
                    else:
                        data['Spectra'][key_ff][sp_type] = np.zeros_like(
                            data["energy_grid"])
            else:
                for i, tok in enumerate(tokens):
                    if i == 0:
                        data["energy_grid"][i_point] = float(tok)
                        continue
                    toks = headers_spectra[i-1].split('(')
                    key_ff = toks[1].strip(')')
                    sp_type = toks[0].strip()

                    data['Spectra'][key_ff][sp_type][i_point] = float(tok)
                i_point += 1

    return data


def main(argv=None):
    parser = ArgumentParser(description='Plot the spectra')
    parser.add_argument('filename', type=str,
                        help='The filename to read', action='store')
    parser.add_argument('--types', nargs="+", default=[], help='The types of spectra to plot.',
                        choices=["Single", "Sum", "Angular", "Alpha"])
    parser.add_argument('--ffs', nargs="+", default=[], help='The Fermi functions to use',
                        choices=["Numeric", "PointLike", "ChargedSphere"])
    args = parser.parse_args()

    data = load_data(args.filename)
    plt.style.use('spectra.mplstyle')
    # Plot the data
    for sp_type in args.types:
        sp_type_raw = ph.SPECTRUM_TYPES[sp_type]
        fig, axs = plt.subplots(nrows=2, ncols=1, figsize=(
            10, 6), height_ratios=[1, .3], sharex=True, gridspec_kw={'wspace': 0, 'hspace': 0})
        for_ratio = []
        for_ratio_titles = []
        for ff_type in args.ffs:
            if sp_type_raw == ph.ALPHASPECTRUM:
                try:
                    dhde = data["PSFs"][ff_type]["H"] * \
                        data["Spectra"][ff_type]["dH/de"]
                    dgde = data["PSFs"][ff_type]["G_single"] * \
                        data["Spectra"][ff_type]["dG/de"]
                    to_plot = dhde / dgde
                except Exception as e:
                    print(f"Error in calculating alpha: {e}")
                    continue

                axs[0].plot(data['energy_grid'], to_plot, label=f'{ff_type}')

            elif ph.SPECTRUM_TYPES_NICE[sp_type_raw] not in data['Spectra'][ff_type]:
                print(
                    f"Type {sp_type} not found in input file for Fermi function {ff_type}. Skipping.")
                continue
            else:
                to_plot = data['Spectra'][ff_type][ph.SPECTRUM_TYPES_NICE[sp_type_raw]]
                axs[0].plot(data['energy_grid'],
                            to_plot,
                            label=f'{ff_type}')

            for_ratio.append(to_plot)
            for_ratio_titles.append(ff_type)

        if len(for_ratio) > 1:
            for i in range(1, len(for_ratio)):
                axs[1].plot(data['energy_grid'], 1-for_ratio[i] / for_ratio[0],
                            label=f'{for_ratio_titles[i]} / {for_ratio_titles[0]}')

        axs[1].set_xlabel(r'E-$m_{e}$ [MeV]')
        axs[0].set_ylabel(f'{ph.SPECTRUM_TYPES_LATEX[sp_type_raw]} [1/MeV]')
        axs[1].set_ylabel(f"Ratio")
        axs[1].set_ylim(-1., 1.)
        axs[1].legend()
        axs[0].legend()

    plt.show()


if __name__ == '__main__':
    main()
