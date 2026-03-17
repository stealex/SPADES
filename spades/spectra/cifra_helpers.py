from numba import njit
import numpy as np
from spades import ph


@njit
def kn(e1: float, e2: float, enu1: float, enu2: float, e1plus: float):
    # print(f"{e1}, {e2}, {enu1}, {enu2}, {e1plus}")
    denum = e1plus**2.0 - (0.5*(e1+enu2-e2-enu1))**2.0
    if (denum < 0):
        print(f"Denum is {denum} for {e1plus} and {e1}, {e2}, {enu1}, {enu2}")
    return e1plus/denum


@njit
def ln(e1: float, e2: float, enu1: float, enu2: float, e1plus: float):
    denum = e1plus**2.0 - (0.5*(e1+enu1-e2-enu2))**2.0
    return e1plus/denum


@njit
def neutrino_integrand_cifra_standard_00(enu1: float, e1: float, e2: float, enu2: float, e1plus: np.ndarray, melems: np.ndarray):
    mk_terms = np.zeros_like(e1plus, dtype=np.float64)
    ml_terms = np.zeros_like(e1plus, dtype=np.float64)
    for i, e1plustmp in enumerate(e1plus):
        m = melems[i]
        mk_terms[i] = m*kn(e1, e2, enu1, enu2, e1plustmp)
        ml_terms[i] = m*ln(e1, e2, enu1, enu2, e1plustmp)

    mk = np.sum(mk_terms)
    ml = np.sum(ml_terms)

    return (1./4.*(np.abs(mk+ml)**2.0)+1./12.*(np.abs(mk-ml)**2.0))*(enu1**2.0)*(enu2**2.0)


@njit
def neutrino_integrand_cifra_standard_02(enu1: float, e1: float, e2: float, enu2: float, e1plus: float) -> float:
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
    k = kn(e1, e2, enu1, enu2, e1plus)
    l = ln(e1, e2, enu1, enu2, e1plus)
    return 3.0*((k-l)**2.0)*(enu1**2.0)*(enu2**2.0)


@njit
def neutrino_integrand_cifra_angular_00(enu1: float, e1: float, e2: float, enu2: float, e1plus: np.ndarray, melems: np.ndarray):
    mk_terms = np.zeros_like(e1plus, dtype=np.float64)
    ml_terms = np.zeros_like(e1plus, dtype=np.float64)
    for i, e1plustmp in enumerate(e1plus):
        m = melems[i]
        mk_terms[i] = m*kn(e1, e2, enu1, enu2, e1plustmp)
        ml_terms[i] = m*ln(e1, e2, enu1, enu2, e1plustmp)

    mk = np.sum(mk_terms)
    ml = np.sum(ml_terms)

    return (1./4.*(np.abs(mk+ml)**2.0)-1./36.*(np.abs(mk-ml)**2.0))*(enu1**2.0)*(enu2**2.0)


@njit
def neutrino_integrand_cifra_angular_02(enu1: float, e1: float, e2: float, enu2: float, e1plus: float) -> float:
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
    k = kn(e1, e2, enu1, enu2, e1plus)
    l = ln(e1, e2, enu1, enu2, e1plus)
    return ((k-l)**2.0)*(enu1**2.0)*(enu2**2.0)
