import numpy as np
from ctypes import cdll, POINTER, c_double, c_int
import os

class RADIALError(Exception):
    pass

_dir_name = os.path.dirname(__file__)
radial_lib = cdll.LoadLibrary(os.path.join(_dir_name,"../build/libradial.so"))

# configuration and wrapper of VINT
radial_lib.vint_.argtypes = [POINTER(c_double), POINTER(c_double),POINTER(c_int)]
radial_lib.vint_.restype = None

def call_vint(r:np.ndarray, rv:np.ndarray)->None:
    """Wrapper function to call VINT from RADIAL

    Args:
        r (np.ndarray): radial points (in atomic units) where the potential was computed
        rv (np.ndarray): values of R*V(R) (in atomic units)
    """
    radial_lib.vint_(r.ctypes.data_as(POINTER(c_double)), 
                     rv.ctypes.data_as(POINTER(c_double)), 
                     c_int(len(r)))
    
# configuration and wrapper of SGRID
radial_lib.sgrid_.argtypes = [POINTER(c_double), # R(1:N)
                              POINTER(c_double), # DR(1:N)
                              POINTER(c_double), # RN(1:N)
                              POINTER(c_double), # R2
                              POINTER(c_double), # DRN
                              POINTER(c_int), # N
                              POINTER(c_int), # NMAX
                              POINTER(c_int), # IER
                              ]
radial_lib.sgrid_.restype = None

def call_sgrid(rn:float, r2:float, drn:float, n:int, nmax:int)->tuple[int,np.ndarray,np.ndarray]:
    """Wrapper function to call SGRID from RADIAL

    Args:
        rn (float): outer grid point (the grid extends from 0 up to RN) >= 1.0E-5
        r2 (float): approximately R(1) (controls grid spacing at small radii). < 1E-2
        drn (float): R(N-1)-R(N-2) (controls grid spacing at large radial distances).
        n (int): tentative number of grid points (it may be increased). See RADIAL documentation
        nmax (int): physical dimention of the grid. n cannot exceed nmax. See RADIAL documentaiton 

    Raises:
        RADIALError: in case SGRID encounters an error 

    Returns:
        tuple[int,np.ndarray,np.ndarray]:(N, R, DR) containing the number of points, the grid and the derivatives of the grid with respect to the index
    """
    ierr=c_int(0)
    n_new = c_int(n)
    r = np.zeros(nmax)
    dr = np.zeros(nmax)
    
    radial_lib.sgrid_(r.ctypes.data_as(POINTER(c_double)),
                      dr.ctypes.data_as(POINTER(c_double)),
                      c_double(rn),
                      c_double(r2),
                      c_double(drn),
                      n_new,
                      c_int(nmax),
                      ierr)
    
    if (ierr.value != 0):
        raise RADIALError(f"SGRID returned the following error: {ierr.value: d}")
    
    return (n_new.value, r[:n_new.value], dr[:n_new.value])

# configuration and wrapper for DBOUND and DERROR
radial_lib.dbound_.argtypes = [POINTER(c_double), # E
                               POINTER(c_double), #EPS
                               POINTER(c_int), # N
                               POINTER(c_int), # K
                               ]
radial_lib.dbound_.restype = None

radial_lib.derror_.argtypes = [POINTER(c_int), # IERR
                                    ]
radial_lib.derror_.restype = None

def call_dbound(energy:float, n:int, k:int, eps:float=1E-14)->float:
    """Wrapper function to call DBOUND from RADIAL

    Args:
        energy (float): trial bound energy in atomic units (<0)
        n (int): principal quantum number (>=1)
        k (int): relativistic quantum number kappa (-n<=kappa<=n-1)
        eps (float, optional): global tolerance, i.e. allowed relative error
            in the summation of the radial function series (>=1E-15). 
            Defaults to 1E-14.

    Raises:
        ValueError: in case n<1
        ValueError: in case k is unphysical
        ValueError: in case of a positive trial energy
        ValueError: in case eps < 1E-15
        RADIALError: in case something went wrong in DBOUND

    Returns:
        float: value of the binding energy obtained
    """
    if (n < 1):
        raise ValueError("n should be >= 1")
    if (k == 0) or (k < -1*n) or (k > n-1):
        raise ValueError("k should be different from 0 and -n <= k <= n-1")
    if (energy > 0):
        raise ValueError("Bound states should have energy < 0")
    if (eps < 1E-15):
        raise ValueError("eps should be larger than 1E-15")
    
    e_new = c_double(energy)
    # call DBOUND
    radial_lib.dbound_(e_new,
                      c_double(eps),
                      c_int(n),
                      c_int(k))
    
    # call DERROR to see if there was a problem
    ierr = c_int(0)
    radial_lib.derror_(ierr)
    if (ierr.value > 0):
        raise RADIALError(f"DBOUND returned the following error: {ierr.value: d}")
    
    return e_new.value

# configuration for DFREE
radial_lib.dfree_.argtypes = [POINTER(c_double), # E
                              POINTER(c_double), # EPS
                              POINTER(c_double), # PHASE
                              POINTER(c_int), # K
                              POINTER(c_int), # IRWF
                              ]
radial_lib.dfree_.restype = None

def call_dfree(energy:float, k:int, eps:float=1E-14)->float:
    """Wrapper function to call DFREE

    Args:
        energy (float): energy of the scattering state in atomic units
        k (int): relativistic quantum number (!= 0)
        eps (float, optional): global tolerance, i.e. allowed relative error
            in the summation of the radial function series (>=1E-15). 
            Defaults to 1E-14.

    Raises:
        ValueError: in case k is unphysical
        ValueError: in case of a negative scattering energy
        ValueError: in case eps < 1E-15
        RADIALError: in case something went wrong in DFREE

    Returns:
        float: value of the inner phase shift (rad)
    """
    if (k == 0):
        raise ValueError("k should be different from 0")
    if (energy < 0):
        raise ValueError("Scattering states should have energy > 0")
    if (eps < 1E-15):
        raise ValueError("eps should be larger than 1E-15")
    
    phase = c_double(0.)
    # call DFREE
    radial_lib.dfree_(c_double(energy),
                      c_double(eps),
                      phase,
                      c_int(k),
                      c_int(1))
    
    # call DERROR to see if there was a problem
    ierr = c_int(0)
    radial_lib.derror_(ierr)
    if (ierr.value > 0):
        raise RADIALError(f"DFREE returned the following error: {ierr.value : d}")
    
    return phase.value

radial_lib.setrgrid_.argtypes = [POINTER(c_double), # R
                                 POINTER(c_int), #N
                                 ]
radial_lib.setrgrid_.restype = None

def call_setrgrid(r:np.ndarray)->None:
    """Sets the intergation grid for RADIAL

    Args:
        r (np.ndarray): points where the wave function has to be computed (atomic units)

    Raises:
        RADIALError: in case something went wrong with SETRGRID
    """
    radial_lib.setrgrid_(r.ctypes.data_as(POINTER(c_double)),
                         c_int(len(r)))
    
    
radial_lib.getpq_.argtypes = [POINTER(c_double),# P
                              POINTER(c_double),# Q,
                              POINTER(c_int), # NGP
                              ]
radial_lib.getpq_.restype = None
def call_getpq(n:int):
    p = np.zeros(n)
    q = np.zeros(n)
    
    radial_lib.getpq_(p.ctypes.data_as(POINTER(c_double)),
                      q.ctypes.data_as(POINTER(c_double)),
                      c_int(n))
    # call DERROR to see if there was a problem
    ierr = c_int(0)
    radial_lib.derror_(ierr)
    if (ierr.value > 0):
        raise RADIALError(f"GETPQ returned the following error: {ierr.value : d}")
    
    return (p,q)
    