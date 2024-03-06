import numpy as np
from wrappers import radial_wrapper

rPoints = np.linspace(0., 100., 1000)
rvValues = -1.*np.ones_like(rPoints)

radial_wrapper.call_vint(rPoints, rvValues)
npoints, rGrid, drGrid = grid_info = radial_wrapper.call_sgrid(100., 1E-4, 1E-1, 2000, 20000)
radial_wrapper.call_setrgrid(rGrid)

for n in range(10):
    for k in range(-n, n):
        if k == 0:
            continue
        e_schrod = -0.5/(n*n)
        e_new = radial_wrapper.call_dbound(e_schrod, n, k)
        print(e_new)