#!/usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from argparse import ArgumentParser
from src import ph, io_handler

# Load the data from the file


def main(argv=None):
    parser = ArgumentParser(description='Plot the spectra')
    parser.add_argument('filename', type=str,
                        help='The filename to read', action='store')
    parser.add_argument('--types', nargs="+", default=[], help='The types of spectra to plot.',
                        choices=["Single", "Sum", "Angular", "Alpha"])
    parser.add_argument('--ffs', nargs="+", default=[], help='The Fermi functions to use',
                        choices=["Numeric", "PointLike", "ChargedSphere"])
    args = parser.parse_args()

    data = io_handler.load_data(args.filename)
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
                axs[1].plot(data['energy_grid'], for_ratio[i] / for_ratio[0],
                            label=f'{for_ratio_titles[i]} / {for_ratio_titles[0]}')

        axs[1].set_xlabel(r'E-$m_{e}$ [MeV]')
        axs[0].set_ylabel(f'{ph.SPECTRUM_TYPES_LATEX[sp_type_raw]} [1/MeV]')
        axs[1].set_ylabel(f"Ratio")
        axs[1].set_ylim(0.5, 1.5)
        axs[1].legend()
        axs[0].legend()

        fig.savefig(f'{sp_type}.png', dpi=300,
                    bbox_inches='tight', transparent=True)

    plt.show()


if __name__ == '__main__':
    main()
