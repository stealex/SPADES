from wrappers import dhfs_wrapper
import yaml
import periodictable
import utils.physics_constants as physics_constants
import numpy as np
import re

class atomic_system:
    def __init__(self, config:dict) -> None:
        self.name = config["name"]
        matches = re.match(r'(\d+)([A-Za-z]+)', self.name)
        self.mass_number = int(matches.group(1))
        self.symbol = matches.group(2)
        
        self.weight = config["weight"]
        if (self.weight < 0.):
            self.weight = float(self.mass_number)
            
        tmp = config["electron_config"]
        if (tmp == "auto"):
            z = periodictable.elements.isotope(self.symbol)
            electron_configuration_filename = os.path.join(os.path.dirname(__file__),
                                                           f"../data/ground_state_config_Z{z:03d}.yaml")
        else:
            electron_configuration_filename = tmp
            
        with open(electron_configuration_filename, 'r') as f:
            config_tmp = yaml.safe_load(f)
           
        self.Z = config_tmp["Z"]
        self.name = config_tmp["Name"]
        self.electron_config = np.array(config_tmp["configuration"])
        
        
        self.n_values = self.electron_config[:,0].astype(int)
        self.l_values = self.electron_config[:,1].astype(int)
        self.jj_values = self.electron_config[:,2].astype(int)
        self.occ_values = self.electron_config[:,3].astype(float)
        
    def print(self):
        text=f"""DHFS configuration:
Z: {self.Z:d}
name: {self.name:s}
configuration:[n,l,2j,occupation]        
"""
        for c in self.electron_config:
            text = text + f"  - [{c[0]:d},{c[1]:d},{c[2]:d},{c[3]:f}]\n"
        
        print(text)
            


class dhfs_handler:
    """
    Class used for handling DHFS calculations and results for a specific system.
    Example usage:
        from utils import dhfs_handler
        
        handler = dhfs_handler(config, label)
        handler.run_dhfs(100, radius_unit='bohr')
        handler.rertrieve_results()
        
    Attributes
    ----------
    label : str
        label for bookkeeping and printing
    dhfs_config: dhfs_configuration
        electron configuration and nuclear charge. See documentation of class dhfs_configuration
    atomic_weight: float
        mass number in g/mol
    rad_grid: np.ndarray
        radial grid where wavefunction are computed
    p_grid: np.ndarray
        values of the P component of the wavefunctions. Dimensions = (nb shells, nb radial points)
    q_grid: np.ndarray
        values of the Q component of the wavefunctions. Dimensions = (nb shells, nb radial points)
    rv_nuc: np.ndarray
        values of the r times electrostatic potential of the nucleus. Dimensions = len(rad_grid)
    rv_el: np.ndarray
        values of the r times electronic potential. Dimensions = len(rad_grid)
    rv_ex: np.ndarray
        values of the r times exchange potential. Dimensions = len(rad_grid)
    """
    
    def __init__(self, config:dict|atomic_system, label:str, atomic_weight:float=-1.0) -> None:
        """Handler class for DHFS.f usage.

        Args:
            configuration_file (dict): dictonary {"name", "label, "electron_config"}
            label (str): Relevant label for calculations.
            atomic_weight (float, optional): Atomic weight in g/mol. If negative, the value from periodictable is used. Defaults to -1
        """
        self.label = label
        if (type(config) == dict):
            self.dhfs_config = atomic_system(config)
        elif (type(config) == atomic_system):
            self.dhfs_config = config
        else:
            raise TypeError("config should be either a dictionary or an object of type atomic_system")
        
        if atomic_weight > 0.:
            self.atomic_weight = atomic_weight
        else:
            self.atomic_weight = periodictable.elements[self.dhfs_config.Z].mass
            
    def print(self):
        print(f"DHFS handler for {self.label}")
        print(f"Atomic weight = {self.atomic_weight:f}")
        self.dhfs_config.print()
        
    def run_dhfs(self, max_radius:float, n_points=1000, radius_units:str="fm"):
        """Organizes the call to DHFS_MAIN from DHFS.f.
        Sets appropriate parameters first.
        

        Args:
            max_radius (float): Maximum radius for radial grid
            n_points (int, optional): Number of radial points to use (tentative). Defaults to 1000.
            radius_units (str, optional): Units for distance. Options: "bohr", "fm". Defaults to "fm".

        Raises:
            ValueError: _description_
        """
        
        dhfs_wrapper.call_configuration_input(self.dhfs_config.n_values, 
                                              self.dhfs_config.l_values, 
                                              self.dhfs_config.jj_values, 
                                              self.dhfs_config.occ_values,
                                              self.dhfs_config.Z)
        if (radius_units == "fm"):
            max_radius = max_radius/physics_constants.bohr_radius
        else:
            if (radius_units != "bohr"):
                raise ValueError(f"Unknown unit {radius_units:s}")
            
        n_tmp = dhfs_wrapper.call_set_parameters(self.atomic_weight, max_radius, n_points)
        self.n_grid_points = n_tmp.value
        
        dhfs_wrapper.call_dhfs_main(1.5)
        
    def retrieve_dhfs_results(self):
        """
    Retrieves results from the DHFS (Dirac-Hartree-Fock-Slater) calculations.

    Calls the DHFS wrapper to obtain wavefunction components (r, p, q) and potential components (vn, vel, vex).

    The wavefunction components are computed based on the electron configuration and grid points specified 
    in dhfs_config and n_grid_points respectively.

    The radial grid 'r' is stored in self.rad_grid, wavefunction components 'p' and 'q' are stored in
    self.p_grid and self.q_grid respectively.

    The potential components 'vn', 'vel', and 'vex' are obtained and stored in self.rv_nuc, self.rv_el, 
    and self.rv_ex respectively.
    """
        (r, p, q) = dhfs_wrapper.call_get_wavefunctions(len(self.dhfs_config.electron_config), self.n_grid_points) 
        self.rad_grid = r
        self.p_grid = p
        self.q_grid = q
        
        vn, vel, vex = dhfs_wrapper.call_get_potentials(self.n_grid_points)
        self.rv_nuc = vn
        self.rv_el = vel
        self.rv_ex = vex
        
        