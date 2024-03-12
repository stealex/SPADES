import yaml
import periodictable
import re
import os
import numpy as np
from utils.dhfs_handler import atomic_system, create_ion

class bound_config:
    def __init__(self, params:dict) -> None:
        self.max_r = params["max_r"]
        if "radius_unit" in params:
            a = params["radius_unit"]
        else:
            a = "bohr"
        
        self.radius_unit = a
        self.n_radial_points = params["n_radial_points"]
        
        if (type(params["n_values"]) == str):
            if (params["n_values"] != "auto"):
                raise ValueError("n_values has to be 'auto', a number or a list")
            else:
                self.n_values = range(1,10)
        elif (type(params["n_values"] == int)):
            self.n_values = range(1,params["n_values"])
        elif (type(params["n_values"]) == list):
            self.n_values = params["n_values"]
        else:
            raise ValueError("Cannot interpret option n_values. Known options: auto, <int>, list[int]")
            
        if (type(params["k_values"]) == str):
            if (params["k_values"] != 'auto'):
                raise ValueError("k_values has to be 'auto', a number or a list")
            else:
                self.k_values = []
                for i_n in self.n_values:
                    k_tmp = []
                    for i_k in range(-i_n, i_n):
                        if (i_k == 0):
                            continue
                        
                        k_tmp.append(i_k)
                    self.k_values.append(k_tmp)
        elif (type(params["k_values"]) == int):
            k_proposed = params["k_values"]
            self.k_values = []
            for i_n in self.n_values:
                if (k_proposed >= i_n) or (k_proposed < -i_n):
                    continue
                self.k_values.append([k_proposed])
        elif (type(params["k_values"]) == list):
            self.k_values = []
            for i_n in self.n_values:
                k_tmp = []
                for i_k in params["k_values"]:
                    if (i_k >= i_n) or (i_k < -i_n):
                        continue
                    k_tmp.append(i_k)
                self.k_values.append(k_tmp)
                
    def print(self):
        print("Configuration for bound states")
        print(f"  - Maximum radial distance: {self.max_r : .3E} {self.radius_unit}")
        print(f"  - Number of radial points: {self.n_radial_points: d}")
        print(f"  - N and K values:")
        for i in range(len(self.n_values)):
            print(f"    - {self.n_values[i]}", self.k_values[i])
        
       
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
    def __init__(self, config:dict) -> None:
        self.task_name = config["task"]
        self.process_name = config["process"]
        self.output_dir = config["output_directory"]
        
        # atoms
        self.initial_atom = atomic_system(config["initial_atom"])
        self.final_atom = create_ion(self.initial_atom, self.initial_atom.Z + 2)
        
        # technicals
        self.bound_config = bound_config(config["bound_states"])
        self.scattering_config = scattering_config(config["scattering_states"])
        
        