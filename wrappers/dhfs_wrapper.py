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
    dhfs_lib.set_parameters_(ctypes.c_double(atomic_weight),
                             ctypes.c_double(outer_radius),
                             ctypes.c_int(n_grid_points))


# configuration and wrapper for DHFS_MAIN
dhfs_lib.dhfs_main_.argtypes = [ctypes.POINTER(ctypes.c_double)]
dhfs_lib.dhfs_main_.restype = None

def call_dhfs_main(alpha:float):
    dhfs_lib.dhfs_main_(ctypes.c_double(alpha))