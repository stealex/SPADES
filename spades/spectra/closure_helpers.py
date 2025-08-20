from numba import njit


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
