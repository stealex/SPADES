import numpy as np
from handlers.wavefunction_handlers import bound_handler, scattering_handler
from numba import njit
from scipy import integrate

class spectrum:
    def __init__(self, q_value:float, energy_grid:float, spectrum_type) -> None:
        self.q_value = q_value
        self.energy_grid = energy_grid
        if (spectrum_type == )
        
    def compute(self, energy_grid:np.ndarray):
        pass
    
    def neutrino_integrant(self, e1:float, e2:float, enu:float):
        pass
    
    def compute_neutrino_integral(self, e1_grid:np.ndarray, e2_grid:np.ndarray, *args):
        pass        
                

class closure_spectrum(spectrum):
    def __init__(self, q_value:float, enei:float) -> None:
        super().__init__(q_value)
        self.enei = enei
    
    def neutrino_integrant(self, e1: float, e2: float, enu: float):
        return _closure_neutrino_integrant_standard(e1,e2,enu,self.q_value, self.enei)
    
    def compute_neutrino_integral(self, e1_grid: np.ndarray, e2_grid: np.ndarray, enei:float):
        return super().compute_neutrino_integral(e1_grid, e2_grid, enei)
    
    def compute(self, energy_grid: np.ndarray):
        # compute neutrino integral
        
    
class spectrum_handler:
    def __init__(self, sh:scattering_handler, bh:bound_handler | None = None) -> None:
        self.scatttering_handler = sh
        self.bound_handler = bh
        
    def compute_bare_spectrum(self):
        