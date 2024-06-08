import numpy as np

from src.wavefunctions import scattering_config, scattering_handler
from src.fermi_functions import numeric, charged_sphere, point_like

from src import ph

import matplotlib.pyplot as plt

z_nuc = 40
r_nuc = 5.0  # fm

rPoints = np.linspace(0., 100., 1000)
rvValues = -z_nuc*np.ones_like(rPoints)

min_ke = 1E-3  # 1 keV
max_ke = 3.0  # 3 MeV
scat_config = scattering_config(
    30., 2000, min_ke/ph.hartree_energy, max_ke/ph.hartree_energy, 100, [-1, 1])
scat_handler = scattering_handler(z_nuc, z_nuc+2, scat_config)
scat_handler.set_potential(rPoints, rvValues)
scat_handler.compute_scattering_states()

ff_numeric = numeric(scat_handler, r_nuc)
ff_pointlike = point_like(z_nuc, r_nuc)
ff_charged_sphere = charged_sphere(z_nuc, r_nuc)

e_grid = np.logspace(np.log10(min_ke), np.log10(max_ke), 100)
ff_numeric_vals = np.zeros_like(e_grid)
ff_pointlike_vals = np.zeros_like(e_grid)
ff_charged_sphere_vals = np.zeros_like(e_grid)
for i_e in range(len(e_grid)):
    ff_numeric_vals[i_e] = ff_numeric.ff0_eval(e_grid[i_e])
    ff_pointlike_vals[i_e] = ff_pointlike.ff0_eval(e_grid[i_e])
    ff_charged_sphere_vals[i_e] = ff_charged_sphere.ff0_eval(e_grid[i_e])


fig, ax = plt.subplots(ncols=2)
ax[0].plot(e_grid, ff_numeric_vals, label="numeric")
ax[0].plot(e_grid, ff_charged_sphere_vals, label='charged sphere')
ax[0].plot(e_grid, ff_pointlike_vals, label="point-like")
ax[0].set_xscale('log')
ax[0].set_yscale('log')

ax[1].plot(e_grid, ff_numeric_vals/ff_pointlike_vals)
ax[1].set_xscale('log')

gm1_numeric = np.zeros_like(e_grid, dtype=complex)
gm1_pointlike = np.zeros_like(e_grid, dtype=complex)
for i_e in range(len(e_grid)):
    gm1_numeric[i_e] = ff_numeric.gm1[i_e]
    gm1_pointlike[i_e] = ff_pointlike.gk(e_grid[i_e], -1)

fig, ax = plt.subplots()
ax.plot(e_grid, np.abs(gm1_numeric))
ax.plot(e_grid, np.abs(gm1_pointlike))


plt.show()
