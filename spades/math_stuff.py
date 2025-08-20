from math import atan
from . import ph
import numpy as np
from scipy.special import gamma, spence
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


def r_radiative(e_total, e_max):
    p = np.sqrt(e_total**2.0-ph.electron_mass**2.0)
    beta = p/e_total
    atanh = np.arctanh(beta)
    return 1.+ph.fine_structure/(2.0*np.pi) *\
        (3.*np.log(ph.proton_mass/ph.electron_mass)-0.75-4./beta*spence(1.-2.*beta/(1.+beta)) +
         atanh/beta*(2.*(1+beta*beta) + (e_max-e_total)**2.0/(6*e_total**2.0) - 4.*atanh) +
         4.*(atanh/beta - 1.)*((e_max-e_total) /
                               (3.*e_total)-1.5+np.log(2*(e_max-e_total)/ph.electron_mass)))
