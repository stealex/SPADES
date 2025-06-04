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


@njit
def neutrino_integrand_closure_standard_02(enu1: float, e1: float, e2: float, enu2: float, enei: float) -> float:
    """Computes the neutrino integrant for standard (i.e. "G") psfs/spectra in closure approximation for the 0+ -> 2+ transition.

    Args:
        enu1 (float): energy of first (anti-)neutrino: over this we integrate
        e1 (float): total energy of first electron (positron)
        e2 (float): total energy of second electron (positron)
        enu2 (float): energy of second (anti-)neutrino. 
        enei (float): <E_N> - E_I

    Returns:
        float: value of integrant
    """
    k = kn(e1, e2, enu1, enu2, enei)
    l = ln(e1, e2, enu1, enu2, enei)
    return 3.0*((k-l)**2.0)*(enu1**2.0)*(enu2**2.0)


@njit
def neutrino_integrand_closure_angular_00(enu1: float, e1: float, e2: float, enu2: float, enei: float) -> float:
    """Computes the neutrino integrant for "Angular" (i.e. H) psfs/spectra in closure approximation for the 0+ -> 2+ transition.

    Args:
        enu1 (float): energy of first (anti-)neutrino: over this we integrate
        e1 (float): total energy of first electron (positron)
        e2 (float): total energy of second electron (positron)
        enu2 (float): energy of second (anti-)neutrino. 
        enei (float): <E_N> - E_I

    Returns:
        float: value of integrant
    """
    k = kn(e1, e2, enu1, enu2, enei)
    l = ln(e1, e2, enu1, enu2, enei)
    return 1./3.*(2*k*k + 2*l*l + 5*k*l)*(enu1**2.0)*(enu2**2.0)


@njit
def neutrino_integrand_closure_angular_02(enu1: float, e1: float, e2: float, enu2: float, enei: float) -> float:
    """Computes the neutrino integrant for "Angular" (i.e. H) psfs/spectra in closure approximation for the 0+ -> 2+ transition.

    Args:
        enu1 (float): energy of first (anti-)neutrino: over this we integrate
        e1 (float): total energy of first electron (positron)
        e2 (float): total energy of second electron (positron)
        enu2 (float): energy of second (anti-)neutrino. 
        enei (float): <E_N> - E_I

    Returns:
        float: value of integrant
    """
    k = kn(e1, e2, enu1, enu2, enei)
    l = ln(e1, e2, enu1, enu2, enei)
    return ((k-l)**2.0)*(enu1**2.0)*(enu2**2.0)


def r_radiative(e_total, e_max):
    p = np.sqrt(e_total**2.0-ph.electron_mass**2.0)
    beta = p/e_total
    atanh = np.arctanh(beta)
    return 1.+ph.fine_structure/(2.0*np.pi) *\
        (3.*np.log(ph.proton_mass/ph.electron_mass)-0.75-4./beta*spence(1.-2.*beta/(1.+beta)) +
         atanh/beta*(2.*(1+beta*beta) + (e_max-e_total)**2.0/(6*e_total**2.0) - 4.*atanh) +
         4.*(atanh/beta - 1.)*((e_max-e_total) /
                               (3.*e_total)-1.5+np.log(2*(e_max-e_total)/ph.electron_mass)))
