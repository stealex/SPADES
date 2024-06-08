import numpy as np
from src import radial_wrapper, math_stuff

rPoints = np.linspace(0., 100., 1000)
rvValues = -1.*np.ones_like(rPoints)

radial_wrapper.call_vint(rPoints, rvValues)
npoints, rGrid, drGrid = grid_info = radial_wrapper.call_sgrid(
    100., 1E-4, 1E-1, 2000, 20000)
radial_wrapper.call_setrgrid(rGrid)

print(f"{'n':2s} {'k':3s} {'e_found': 10s} {'e_diract': 10s} {'delta_e':}")
for n in range(4):
    for k in range(-n, n):
        if k == 0:
            continue
        e_schrod = -0.5/(n*n)
        e_new = radial_wrapper.call_dbound(e_schrod, n, k)
        e_dirac = math_stuff.hydrogenic_binding_energy(1, n, k)
        print(n, k, e_new, e_dirac, e_new-e_dirac)
