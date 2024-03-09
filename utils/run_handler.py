import yaml
import periodictable
import re
import os
import numpy as np
from dhfs_handler import atomic_system

class bound_config:
    def __init__(self, params:dict) -> None:
        self.max_r = params["max_r"]
        if "radius_unit" in params:
            a = params["radius_unit"]
        else:
            a = "bohr"
        
        self.radius_unit = a
        self.n_radial_points = params["n_radial_points"]
        
        
class scattering_config:
    def __init__(self, params:dict) -> None:
        self.max_r = params["max_r"]
        if "radius_unit" in params:
            a = params["radius_unit"]
            if (a != "bohr") and (a != "fm"):
                raise ValueError("radius_unit can be either absent, 'fm' or 'bohr'")
        else:
            a = "bohr"
        
        self.radius_unit = a
        self.n_radial_points = params["n_radial_points"]
        
        self.min_ke = params["min_ke"]
        self.max_ke = params["max_ke"]
        
        if "ke_unit" in params:
            a = params["ke_unit"]
            if (a != 'MeV' and (a != 'hartree')):
                raise ValueError("ke_unit can be either absent, 'MeV' or 'hartree'")
        else:
            a = "MeV"
        self.ke_unit = a
        
        self.n_ke_points = params["n_ke_points"]
        self.ke_grid_type = params["ke_grid_type"]
        if (self.ke_grid_type != 'lin') and (self.ke_grid_type != 'log'):
            raise ValueError('ke_grid_type can be either "lin" or "log"')
 
        
class run_input:
    def __init__(self, config_file_name:str) -> None:
        config = yaml.safe_load(config_file_name)
        
        self.task_name = config["task"]
        self.process_name = config["process"]
        self.output_dir = config["output_directory"]
        
        # atoms
        self.initial_atom = atomic_system({"name": config["initial_atom"]["name"],
                                           "weight": config["initial_atom"]["weight"],
                                           "electron_config": config["initial_atom"]["electron_config"]})
        
        final_atom_dict = {"name": config["final_atom"]["name"],
                           "weight": config["final_atom"]["weight"],
                           "electron_config": config["final_atom"]["electron_config"]}
        if (final_atom_dict["name"] == "auto"):
            final_atom_dict["name"] = periodictable.elements[self.initial_atom.Z+2].symbol
        self.final_atom = atomic_system(final_atom_dict)
        
        # technicals
        self.bound_config = bound_config(config["bound_states"])
        self.scattering_config = scattering_config(config["scattering_config"])
        
        