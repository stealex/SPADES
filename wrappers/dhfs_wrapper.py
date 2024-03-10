import numpy as np
from ctypes import cdll
import ctypes
import os

try:
    _dir_name = os.path.dirname(__file__)
    dhfs_lib = cdll.LoadLibrary(os.path.join(_dir_name,"../build/libdhfs.so"))
except OSError as e:
    print(f"Error: Unable to load dhfs.so: {e}")

# configuration and wrapper for CONFIGURATION_INPUT
dhfs_lib.configuration_input_.argtypes = [ctypes.POINTER(ctypes.c_int),
                                          ctypes.POINTER(ctypes.c_int),
                                          ctypes.POINTER(ctypes.c_int),
                                          ctypes.POINTER(ctypes.c_double),
                                          ctypes.POINTER(ctypes.c_int),
                                          ctypes.POINTER(ctypes.c_int)]
dhfs_lib.configuration_input_.restype = None

def call_configuration_input(n:np.ndarray, l:np.ndarray, jj:np.ndarray, occup:np.ndarray, i_z:int):

    dhfs_lib.configuration_input_(n.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
                                  l.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
                                  jj.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
                                  occup.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
                                  ctypes.c_int(i_z),
                                  ctypes.c_int(len(n)))

# configuration and wrapper for SET_PARAMETERS
dhfs_lib.set_parameters_.argtypes = [ctypes.POINTER(ctypes.c_double),
                                     ctypes.POINTER(ctypes.c_double),
                                     ctypes.POINTER(ctypes.c_int)]
dhfs_lib.set_parameters_.restype = None

def call_set_parameters(atomic_weight:float, outer_radius:float, n_grid_points:int):
    np = ctypes.c_int(n_grid_points)
    dhfs_lib.set_parameters_(ctypes.c_double(atomic_weight),
                             ctypes.c_double(outer_radius),
                             np)
    return np


# configuration and wrapper for DHFS_MAIN
dhfs_lib.dhfs_main_.argtypes = [ctypes.POINTER(ctypes.c_double)]
dhfs_lib.dhfs_main_.restype = None

def call_dhfs_main(alpha:float):
    dhfs_lib.dhfs_main_(ctypes.c_double(alpha))
    
# configuration and wrapper for GET_WAVEFUNCTIONS
dhfs_lib.get_wavefunctions_.argtypes = [ctypes.POINTER(ctypes.c_double),
                                        ctypes.POINTER(ctypes.c_double),
                                        ctypes.POINTER(ctypes.c_double),
                                        ctypes.POINTER(ctypes.c_int),
                                        ctypes.POINTER(ctypes.c_int)]
dhfs_lib.get_wavefunctions_.restype = None
def call_get_wavefunctions(n_shells:int, n_points:int):
    rad_grid = np.zeros(n_points)
    p_grid = np.zeros((n_shells, n_points), order="F")
    q_grid = np.zeros((n_shells, n_points), order="F")
    
    dhfs_lib.get_wavefunctions_(rad_grid.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
                                p_grid.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
                                q_grid.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
                                ctypes.c_int(n_shells),
                                ctypes.c_int(n_points))
    return (rad_grid, p_grid, q_grid)
    
# configuration and wrapper for GET_POTENTIALS
dhfs_lib.get_potentials_.argtypes = [ctypes.POINTER(ctypes.c_double),
                                     ctypes.POINTER(ctypes.c_double),
                                     ctypes.POINTER(ctypes.c_double),
                                     ctypes.POINTER(ctypes.c_int)]
dhfs_lib.get_potentials_.restype = None
def call_get_potentials(n_points:int):
    v_nuc = np.zeros(n_points)
    v_el = np.zeros(n_points)
    v_ex = np.zeros(n_points)
    
    dhfs_lib.get_potentials_(v_nuc.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
                             v_el.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
                             v_ex.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
                             ctypes.c_int(n_points))
    return (v_nuc, v_el, v_ex)

# wrapper and definition for GET_BINDING_ENERGIES
dhfs_lib.get_binding_energies_.argtypes = [ctypes.POINTER(ctypes.c_double),
                                           ctypes.POINTER(ctypes.c_int)]
dhfs_lib.get_binding_energies.restype = None
def call_get_binding_energies(n_shells:int) -> np.ndarray:
    energies = np.zeros(n_shells)
    dhfs_lib.get_binding_energies(energies.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
                                  ctypes.c_int(n_shells))
    return energies
    