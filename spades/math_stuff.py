from . import ph
import numpy as np
from scipy.special import gamma
from numba import njit


def hydrogenic_binding_energy(z: int, n: int, k: int):
    total_energy = ph.electron_mass*np.power(
        1. + (z*ph.fine_structure/(n-np.abs(k) + np.sqrt(k*k-np.power(z*ph.fine_structure, 2.))))**2.0, -1./2.)
    return (total_energy - ph.electron_mass)


def dirac_gamma(kappa: float, z: float):
    return np.sqrt(np.abs(kappa)**2 - ((ph.fine_structure*z)**2.0))


def dirac_nu(energy: float, z: float, kappa: int):
    x = ph.fine_structure*z*(energy+2.0*ph.electron_mass)
    y = -1.0*(kappa+dirac_gamma(kappa, z)) * \
        np.sqrt(energy*(energy+2.0*ph.electron_mass))
    return np.arctan2(y, x)


def sommerfeld_param(z1, z2, energy):
    return ph.fine_structure*z1*z2*(energy+ph.electron_mass)/np.sqrt(energy*(energy+2.0*ph.electron_mass))


def coulomb_phase_shift(energy: float, z_inf: float, kappa: int):
    sinf = 1 if ((z_inf < 0) and kappa < 0) else 0
    l = kappa if kappa > 0 else -kappa-1
    nu = dirac_nu(energy, z_inf, kappa)
    dirac_gam = dirac_gamma(kappa, z_inf)
    eta = sommerfeld_param(z_inf, 1., energy)
    gam = gamma(dirac_gam+1.0j*eta)
    arg_gamma = np.arctan2(np.imag(gam), np.real(gam))
    result = nu-(dirac_gam-1-l)*np.pi/2. + arg_gamma - sinf*np.pi
    return result


@njit
def kn(e1: float, e2: float, enu1: float, enu2: float, enei: float):
    return 1./(e1+enu1+enei) + 1./(e2+enu2+enei)


@njit
def ln(e1: float, e2: float, enu1: float, enu2: float, enei: float):
    return 1./(e1+enu2+enei) + 1./(e2+enu1+enei)


@njit
def neutrino_integrand_closure_standard_00(enu1: float, e1: float, e2: float, enu2: float, enei: float):
    k = kn(e1, e2, enu1, enu2, enei)
    l = ln(e1, e2, enu1, enu2, enei)
    return (k*k+l*l+k*l)*(enu1**2.0)*(enu2**2.0)
