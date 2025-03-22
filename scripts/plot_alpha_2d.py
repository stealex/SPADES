import os
import yaml
from argparse import ArgumentParser
from src import ph, fermi_functions, exchange, spectra
from src.wavefunctions import wavefunctions_handler, bound_config, scattering_config
from src.dhfs import atomic_system, create_ion
from src import io_handler
import matplotlib.pyplot as plt
import numpy as np
from src.spectra import spectra_config
from src import radial_wrapper
import time

from matplotlib import cm, rcParams


def fmt(x):
    s = f"{x:.1f}"
    if s.endswith("0"):
        s = f"{x:.0f}"
    return rf"{s} \%" if plt.rcParams["text.usetex"] else f"{s} %"


textsize = 20
ax_textsize = 20
plotw = 16  # 10+16
ploth = 6

default_palette = "RdYlBu_r"

plot_lab = ''

plt.rcParams.update(
    {
        "axes.labelsize": 25,
        "axes.titlesize": 25,
        "xtick.labelsize": 25,
        "ytick.labelsize": 25,
        "xtick.top": True,
        "xtick.bottom": True,
        "xtick.direction": "in",
        "ytick.left": True,
        "ytick.right": True,
        "ytick.direction": "in",
        "xtick.major.size": 8,
        "xtick.major.width": 2,
        "xtick.minor.size": 5,
        "xtick.minor.width": 2,
        "xtick.color": 'black',
        "ytick.major.size": 8,
        "ytick.major.width": 2,
        "ytick.minor.size": 5,
        "ytick.minor.width": 2,
        "lines.linewidth": 2,
        'legend.fontsize': 25,

    }
)
plt.rcParams['text.usetex'] = True

plt.close("all")


# data = io_handler.load_2d_spectra("Universe/100Mo_2nu_nops_spectra_2d.dat")
# norm = -2.2703867580972E-18/3.3099112987216E-18
# name = "alpha_2d_nops"

# data = io_handler.load_2d_spectra("100Mo_2nu_withps_spectra_2d.dat")
# norm = data["PSFs"]["Numeric"]["H"]/data["PSFs"]["Numeric"]["G_single"]
# name = "alpha_2d_withps"

# positions = [
#     {"xy": (0.65, 0.65),   "text": -0.9, "rot": -45.},
#     {"xy": (0.24, 0.24), "text": -0.7, "rot": -45.},
#     {"xy": (0.06, 0.06), "text": -0.3, "rot": -45.},
#     {"xy": (0.018, 0.018), "text": -0.1, "rot": -45.},
#     {"xy": (3E-3, 3E-3), "text": -0.01, "rot": -45.},
#     {"xy": (2.E-4, 2.E-4), "text": -0.01, "rot": -45.},
#     {"xy": (6E-3, 5E-4), "text": 0.01, "rot": 45.},
#     {"xy": (5E-4, 6E-3), "text": 0.01, "rot": 45.},
#     {"xy": (0.2, 1.5E-4), "text": 0.1, "rot": 45.},
#     {"xy": (1.5E-4, 0.2), "text": 0.1, "rot": 45.},
# ]

data = io_handler.load_2d_spectra("100Mo_2nu_nops_spectra_2d.dat")
norm = data["PSFs"]["Numeric"]["H"]/data["PSFs"]["Numeric"]["G_single"]
name = "alpha_2d_nops"

positions = [
    {"xy": (0.6, 0.6),   "text": -0.9, "rot": -45.},
    {"xy": (0.2, 0.2), "text": -0.7, "rot": -45.},
    {"xy": (0.04, 0.04), "text": -0.3, "rot": -45.},
    {"xy": (0.004, 0.004), "text": -0.1, "rot": -45.}
]

# data = io_handler.load_2d_spectra("100Mo_2nu_withps_spectra_2d.dat")
# norm = -2.2515762715759E-18/3.3194248682877E-18

alpha_2d = norm*data["Spectra"]["Numeric"]["H"] / \
    data["Spectra"]["Numeric"]["G_single"]

e1 = data["e1_grid"] + data["emin"]
e2 = data["e2_grid"] + data["emin"]

mask = ((e1+e2) <= 3.0344)

# alpha_2d = np.where(mask, alpha_2d, -1.)
# e1 = np.where(mask, e1, 0.)
# e2 = np.where(mask, e2, 0.)


levels = [-0.9, -0.7, -0.3, -0.1, -0.01, 0.01, 0.1]
# fig, ax = plt.subplots(figsize=(10, 10))
# cf = ax.contourf(e1, e2, alpha_2d,
#                  levels=levels,
#                  extend='both',
#                  cmap=cm.Blues_r)
# cs = ax.contour(cf, colors='k')
# ax.clabel(cs,
#           cs.levels,
#           fontsize=20,
#           manual=False,
#           )

# ax.set_xlabel(r"$E_{e_{1}} - m_{e}\:[\textrm{MeV}]$")
# ax.set_ylabel(r"$E_{e_{2}} - m_{e}\:[\textrm{MeV}]$")

# fig.savefig(name+".png", dpi=300,
#             bbox_inches='tight', transparent=True)

fig, ax = plt.subplots(figsize=(10, 10))
cf = ax.contourf(e1, e2, alpha_2d,
                 levels=levels,
                 extend='both',
                 cmap=cm.Blues_r)
cs = ax.contour(cf, colors='k')

for i_pos in range(len(positions)):
    ax.text(positions[i_pos]["xy"][0], positions[i_pos]
            ["xy"][1], positions[i_pos]["text"],
            rotation=positions[i_pos]["rot"],
            fontsize=25)

ax.set_xscale('log')
ax.set_yscale('log')
ax.set_xlabel(r"$E_{e_{1}} - m_{e}\:[\textrm{MeV}]$")
ax.set_ylabel(r"$E_{e_{2}} - m_{e}\:[\textrm{MeV}]$")

fig.savefig(name+"_log.eps", dpi=300,
            bbox_inches='tight', transparent=True)

plt.show()
fig, ax = plt.subplots()
cf = ax.contourf(e1_plot, e2_plot, data["Spectra"]["Numeric"]["dH/de"])
cs = ax.contour(cf, colors='k')
ax.clabel(cs,
          cs.levels,
          fontsize=25,
          manual=False,
          )

fig, ax = plt.subplots()
cf = ax.contourf(e1_plot, e2_plot, data["Spectra"]["Numeric"]["dG/de"])
cs = ax.contour(cf, colors='k')
ax.clabel(cs,
          cs.levels,
          fontsize=20,
          manual=False,
          )

plt.show()
