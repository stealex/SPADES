import numpy as np
from wrappers import radial_wrapper

rPoints = np.linspace(0., 1000., 1000)
rvValues = -1.*np.ones_like(rPoints)

print("calling vint")
radial_wrapper.call_vint(rPoints, rvValues)
npoints, rGrid, drGrid = grid_info = radial_wrapper.call_sgrid(200., 1E-7, 0.5, 1000, 2000)
print("calling setrgrid")
radial_wrapper.call_setrgrid(rGrid)
print("loop over bound states")
for n in range(10):
    for k in range(-n, n):
        if k == 0:
            continue
        e_schrod = -0.5/(n*n)
        e_new = radial_wrapper.call_dbound(e_schrod, n, k, eps=1E-14)
        print(e_new)