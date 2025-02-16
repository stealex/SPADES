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


def g_spec(s_0, s_2, s_22, s_4, xi_31, xi_51):
    return s_0 + xi_31*s_2 + (1./3.)*s_22*(xi_31**2.0) + (1./3. * (xi_31**2.0) + xi_51)*s_4


def h_spec(s_0, s_2, s_22, s_4, xi_31, xi_51):
    return s_0 + xi_31*s_2 + (5./9.)*s_22*(xi_31**2.0) + (2./9. * (xi_31**2.0) + xi_51)*s_4


def alpha_func(spectra, xi_31, xi_51):

    g_0 = spectra["G_single_0"]
    g_2 = spectra["G_single_2"]
    g_22 = spectra["G_single_22"]
    g_4 = spectra["G_single_4"]
    h_0 = spectra["H_0"]
    h_2 = spectra["H_2"]
    h_22 = spectra["H_22"]
    h_4 = spectra["H_4"]
    g = g_spec(g_0, g_2, g_22, g_4, xi_31, xi_51)
    h = h_spec(h_0, h_2, h_22, h_4, xi_31, xi_51)
    return h/g


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


# data = io_handler.load_2d_spectra(
#     "Universe_Taylor/100Mo_2nu_withps_spectra_2d.dat")
# name = "alpha_2d_withps_taylor"


# positions = [
#     {"xy": (0.55, 0.55),   "text": r"$-0.9$", "rot": -45.},
#     {"xy": (0.20, 0.20), "text": r"$-0.7$", "rot": -45.},
#     {"xy": (0.05, 0.05), "text": r"$-0.3$", "rot": -45.},
#     {"xy": (0.015, 0.015), "text": r"$-0.1$", "rot": -45.},
#     {"xy": (2.5E-3, 2.5E-3), "text": r"$-0.01$", "rot": -45.},
#     {"xy": (2.E-4, 2.E-4), "text": r"$-0.01$", "rot": -45.},
#     {"xy": (6E-3, 5E-4), "text":  r"$0.01$", "rot": 45.},
#     {"xy": (5E-4, 6E-3), "text":  r"$0.01$", "rot": 45.},
#     {"xy": (0.2, 1.5E-4), "text": r"$0.1$", "rot": 45.},
#     {"xy": (1.5E-4, 0.2), "text": r"$0.1$", "rot": 45.},
# ]

data = io_handler.load_2d_spectra(
    "Universe_Taylor/100Mo_2nu_nops_spectra_2d.dat")
name = "alpha_2d_nops_taylor"

positions = [
    {"xy": (0.5, 0.5),     "text": r"$-0.9$", "rot": -45.},
    {"xy": (0.18, 0.18),     "text": r"$-0.7$", "rot": -45.},
    {"xy": (0.03, 0.03),   "text": r"$-0.3$", "rot": -45.},
    {"xy": (0.004, 0.004), "text": r"$-0.1$", "rot": -45.}
]

# data = io_handler.load_2d_spectra("100Mo_2nu_withps_spectra_2d.dat")
# norm = -2.2515762715759E-18/3.3194248682877E-18

specs = data["Spectra"]["Numeric"]
for key in specs:
    specs[key] = specs[key] * data["PSFs"]["Numeric"][key]

alpha_2d = alpha_func(specs, 0.45, 0.367*0.45)
print(alpha_2d)

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
