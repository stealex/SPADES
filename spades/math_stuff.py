"""Mathematical helpers for relativistic wavefunctions and radiative terms."""

from math import atan
from . import ph
import numpy as np
from scipy.special import gamma, spence
from numba import njit


def hydrogenic_binding_energy(z: int, n: int, k: int):
    """Approximate Dirac hydrogenic binding energy for a ``(n, kappa)`` shell.

    Parameters
    ----------
    z:
        Nuclear charge.
    n:
        Principal quantum number.
    k:
        Relativistic angular quantum number ``kappa``.

    Returns
    -------
    float
        Binding energy in MeV (negative for bound states).
    """
    total_energy = ph.electron_mass*np.power(
        1. + (z*ph.fine_structure/(n-np.abs(k) + np.sqrt(k*k-np.power(z*ph.fine_structure, 2.))))**2.0, -1./2.)
    return (total_energy - ph.electron_mass)


def dirac_gamma(kappa: float, z: float):
    """Return the relativistic angular factor ``sqrt(kappa^2 - (alpha Z)^2)``.

    Parameters
    ----------
    kappa:
        Relativistic angular quantum number.
    z:
        Effective charge used in the asymptotic potential.

    Returns
    -------
    float
        Dirac ``gamma`` factor.
    """
    return np.sqrt(np.abs(kappa)**2 - ((ph.fine_structure*z)**2.0))


def dirac_nu(energy: float, z: float, kappa: int):
    """Return the Dirac phase contribution used in Coulomb phase shifts.

    Parameters
    ----------
    energy:
        Electron kinetic energy in MeV.
    z:
        Effective asymptotic charge.
    kappa:
        Relativistic angular quantum number.

    Returns
    -------
    float
        Phase contribution ``nu`` in radians.
    """
    x = ph.fine_structure*z*(energy+2.0*ph.electron_mass)
    y = -1.0*(kappa+dirac_gamma(kappa, z)) * \
        np.sqrt(energy*(energy+2.0*ph.electron_mass))
    return np.arctan2(y, x)


def sommerfeld_param(z1, z2, energy):
    """Return Sommerfeld parameter for two charges at electron kinetic energy ``energy``.

    Parameters
    ----------
    z1, z2:
        Charge factors.
    energy:
        Electron kinetic energy in MeV.

    Returns
    -------
    float
        Sommerfeld parameter ``eta``.
    """
    return ph.fine_structure*z1*z2*(energy+ph.electron_mass)/np.sqrt(energy*(energy+2.0*ph.electron_mass))


def coulomb_phase_shift(energy: float, z_inf: float, kappa: int):
    """Compute asymptotic Coulomb phase shift for relativistic partial wave ``kappa``.

    Parameters
    ----------
    energy:
        Electron kinetic energy in MeV.
    z_inf:
        Effective asymptotic charge extracted from the potential tail.
    kappa:
        Relativistic angular quantum number.

    Returns
    -------
    float
        Coulomb phase shift in radians.
    """
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
    """Compute Sirlin-like radiative correction factor ``R`` for beta spectra.

    Parameters
    ----------
    e_total:
        Total electron energy in MeV.
    e_max:
        Endpoint total electron energy in MeV.

    Returns
    -------
    float
        Multiplicative radiative-correction factor.
    """
    p = np.sqrt(e_total**2.0-ph.electron_mass**2.0)
    beta = p/e_total
    atanh = np.arctanh(beta)
    return 1.+ph.fine_structure/(2.0*np.pi) *\
        (3.*np.log(ph.proton_mass/ph.electron_mass)-0.75-4./beta*spence(1.-2.*beta/(1.+beta)) +
         atanh/beta*(2.*(1+beta*beta) + (e_max-e_total)**2.0/(6*e_total**2.0) - 4.*atanh) +
         4.*(atanh/beta - 1.)*((e_max-e_total) /
                               (3.*e_total)-1.5+np.log(2*(e_max-e_total)/ph.electron_mass)))
