from utils import ph
import numpy as np
from utils.ph import fine_structure
from scipy.special import gamma


def hydrogenic_binding_energy(z: int, n: int, k: int):
    total_energy = ph.electron_mass*np.power(
        1. + (z*fine_structure/(n-np.abs(k) + np.sqrt(k*k-np.power(z*fine_structure, 2.))))**2.0, -1./2.)
    return (total_energy - ph.electron_mass)/ph.hartree_energy


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
    # print("sinf ", sinf)
    l = kappa if kappa > 0 else -kappa-1
    # print("l ", l)
    nu = dirac_nu(energy, z_inf, kappa)
    # print("nu ", nu)
    dirac_gam = dirac_gamma(kappa, z_inf)
    # print("dirac_gam ", dirac_gam)
    eta = sommerfeld_param(z_inf, 1., energy)
    # print("eta ", eta)
    gam = gamma(dirac_gam+1.0j*eta)
    # print("gam ", gam)
    arg_gamma = np.arctan2(np.imag(gam), np.real(gam))
    # print("arg_gamma ", arg_gamma)

    return nu-(dirac_gam-1-l)*np.pi/2. + arg_gamma - sinf*np.pi
