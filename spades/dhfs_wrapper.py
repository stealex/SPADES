"""ctypes bindings to the Fortran DHFS library."""

import numpy as np
from ctypes import cdll
import ctypes
import os
from . import ph

def _load_dhfs_library():
    """Load the shared DHFS library from the local build directory.

    Returns
    -------
    ctypes.CDLL
        Loaded ``libdhfs.so`` handle.
    """
    _dir_name = os.path.dirname(__file__)
    return cdll.LoadLibrary(os.path.join(_dir_name, "../build/libdhfs.so"))


def _configure_dhfs_signatures(lib) -> None:
    """Configure ctypes signatures for exported DHFS Fortran routines.

    Parameters
    ----------
    lib:
        Loaded shared-library object returned by :func:`_load_dhfs_library`.
    """
    # configuration and wrapper for CONFIGURATION_INPUT
    lib.configuration_input_.argtypes = [ctypes.POINTER(ctypes.c_int),
                                         ctypes.POINTER(ctypes.c_int),
                                         ctypes.POINTER(ctypes.c_int),
                                         ctypes.POINTER(ctypes.c_double),
                                         ctypes.POINTER(ctypes.c_int),
                                         ctypes.POINTER(ctypes.c_int),
                                         ctypes.POINTER(ctypes.c_int)]
    lib.configuration_input_.restype = None

    # configuration and wrapper for SET_PARAMETERS
    lib.set_parameters_.argtypes = [ctypes.POINTER(ctypes.c_double),
                                    ctypes.POINTER(ctypes.c_double),
                                    ctypes.POINTER(ctypes.c_int),
                                    ctypes.POINTER(ctypes.c_int)]
    lib.set_parameters_.restype = None

    # configuration and wrapper for DHFS_MAIN
    lib.dhfs_main_.argtypes = [ctypes.POINTER(ctypes.c_double),
                               ctypes.POINTER(ctypes.c_int)]
    lib.dhfs_main_.restype = None

    # configuration and wrapper for GET_WAVEFUNCTIONS
    lib.get_wavefunctions_.argtypes = [ctypes.POINTER(ctypes.c_double),
                                       ctypes.POINTER(ctypes.c_double),
                                       ctypes.POINTER(ctypes.c_double),
                                       ctypes.POINTER(ctypes.c_int),
                                       ctypes.POINTER(ctypes.c_int)]
    lib.get_wavefunctions_.restype = None

    # configuration and wrapper for GET_POTENTIALS
    lib.get_potentials_.argtypes = [ctypes.POINTER(ctypes.c_double),
                                    ctypes.POINTER(ctypes.c_double),
                                    ctypes.POINTER(ctypes.c_double),
                                    ctypes.POINTER(ctypes.c_int)]
    lib.get_potentials_.restype = None

    # wrapper and definition for GET_BINDING_ENERGIES
    lib.get_binding_energies_.argtypes = [ctypes.POINTER(ctypes.c_double),
                                          ctypes.POINTER(ctypes.c_int)]
    lib.get_binding_energies_.restype = None


try:
    dhfs_lib = _load_dhfs_library()
    _configure_dhfs_signatures(dhfs_lib)
except OSError as e:
    raise OSError(
        "Unable to load libdhfs.so. Build the Fortran library first "
        "(e.g. via the package build backend)."
    ) from e


def call_configuration_input(n: np.ndarray, l: np.ndarray, jj: np.ndarray, occup: np.ndarray, i_z: int):
    """Pass shell configuration arrays to DHFS.

    Parameters
    ----------
    n, l, jj:
        Arrays of principal/orbital/doubled-total-angular-momentum quantum numbers.
    occup:
        Occupation numbers per shell.
    i_z:
        Nuclear charge ``Z``.
    """
    dhfs_lib.configuration_input_(n.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
                                  l.ctypes.data_as(
                                      ctypes.POINTER(ctypes.c_int)),
                                  jj.ctypes.data_as(
                                      ctypes.POINTER(ctypes.c_int)),
                                  occup.ctypes.data_as(
                                      ctypes.POINTER(ctypes.c_double)),
                                  ctypes.c_int(i_z),
                                  ctypes.c_int(len(n)),
                                  ctypes.c_int(ph.verbose))




def call_set_parameters(atomic_weight: float, outer_radius: float, n_grid_points: int):
    """Configure DHFS radial domain and return adjusted number of grid points.

    Parameters
    ----------
    atomic_weight:
        Atomic weight in g/mol.
    outer_radius:
        Maximum radial distance in DHFS internal units.
    n_grid_points:
        Requested number of grid points.

    Returns
    -------
    int
        Effective number of grid points used by DHFS.
    """
    n_points = ctypes.c_int(n_grid_points)
    dhfs_lib.set_parameters_(ctypes.c_double(atomic_weight),
                             ctypes.c_double(outer_radius),
                             n_points,
                             ctypes.c_int(ph.verbose))
    return n_points.value




def call_dhfs_main(alpha: float):
    """Run the main DHFS solver routine.

    Parameters
    ----------
    alpha:
        Fine-structure-like input parameter expected by the Fortran routine.
    """
    dhfs_lib.dhfs_main_(ctypes.c_double(alpha), ctypes.c_int(ph.verbose))




def call_get_wavefunctions(n_shells: int, n_points: int):
    """Retrieve radial grid and bound-state components from DHFS.

    Parameters
    ----------
    n_shells:
        Number of electron shells.
    n_points:
        Number of radial grid points.

    Returns
    -------
    tuple[np.ndarray, np.ndarray, np.ndarray]
        Radial grid and ``p/q`` wavefunction arrays with shape ``(n_shells, n_points)``.
    """
    rad_grid = np.zeros(n_points)
    p_grid = np.zeros((n_shells, n_points), order="F")
    q_grid = np.zeros((n_shells, n_points), order="F")

    dhfs_lib.get_wavefunctions_(rad_grid.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
                                p_grid.ctypes.data_as(
                                    ctypes.POINTER(ctypes.c_double)),
                                q_grid.ctypes.data_as(
                                    ctypes.POINTER(ctypes.c_double)),
                                ctypes.c_int(n_shells),
                                ctypes.c_int(n_points))
    # p and q have different units for bound and scattering states
    return (rad_grid, p_grid, q_grid)




def call_get_potentials(n_points: int):
    """Retrieve nuclear, electronic, and exchange DHFS potentials.

    Parameters
    ----------
    n_points:
        Number of radial grid points.

    Returns
    -------
    tuple[np.ndarray, np.ndarray, np.ndarray]
        ``(v_nuc, v_el, v_ex)`` arrays on the DHFS radial grid.
    """
    v_nuc = np.zeros(n_points)
    v_el = np.zeros(n_points)
    v_ex = np.zeros(n_points)

    dhfs_lib.get_potentials_(v_nuc.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
                             v_el.ctypes.data_as(
                                 ctypes.POINTER(ctypes.c_double)),
                             v_ex.ctypes.data_as(
                                 ctypes.POINTER(ctypes.c_double)),
                             ctypes.c_int(n_points))
    return (v_nuc,
            v_el,
            v_ex)




def call_get_binding_energies(n_shells: int) -> np.ndarray:
    """Retrieve shell binding energies from DHFS.

    Parameters
    ----------
    n_shells:
        Number of electron shells.

    Returns
    -------
    np.ndarray
        Binding energies for each shell in DHFS internal units.
    """
    energies = np.zeros(n_shells)
    dhfs_lib.get_binding_energies_(energies.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
                                   ctypes.c_int(n_shells))
    return energies
