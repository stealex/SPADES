from utils.physics_constants import electron_mass, hartree
import numpy as np
from utils.physics_constants import fine_structure

def hydrogenic_binding_energy(z:int, n: int, k:int):
    total_energy = electron_mass/hartree * np.power(1. + (z*fine_structure/(n-np.abs(k) + np.sqrt(k*k-np.power(z*fine_structure, 2.))))**2.0, -1./2.)
    return total_energy - electron_mass/hartree