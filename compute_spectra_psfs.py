#!/usr/bin/env python
import os
import yaml
from argparse import ArgumentParser
from utils.run_handler import run_input
from utils.dhfs_handler import dhfs_handler
import numpy as np
from utils import physics_constants

from wrappers import radial_wrapper
from wrappers.radial_wrapper import RADIALError

import matplotlib.pyplot as plt

def main(argv=None):
    '''Command line arguments'''
    parser = ArgumentParser(description="Compute spectra and PSFs for 2nubb decay")
    parser.add_argument("config_file",
                        help="path to yaml configuration file")
    
    args = parser.parse_args()

    with open(args.config_file, 'r') as f:
        run_config = yaml.safe_load(f)
    
    input_config = run_input(run_config)
    
    # bound states of initial atom
    handler_initial = dhfs_handler(input_config.initial_atom, "initial_atom")
    handler_initial.run_dhfs(input_config.bound_config.max_r,
                             input_config.bound_config.n_radial_points, 
                             input_config.bound_config.radius_unit)
    handler_initial.retrieve_dhfs_results()
    rv_modified_initial = handler_initial.build_modified_potential()
    
    # bound states of final atom
    handler_final = dhfs_handler(input_config.final_atom, "final_atom")
    handler_final.run_dhfs(input_config.bound_config.max_r,
                           input_config.bound_config.n_radial_points, 
                           input_config.bound_config.radius_unit)
    handler_final.retrieve_dhfs_results()
    
    # build modified potentials
    rv_modified_final = handler_final.build_modified_potential()
    
    # solve bound states in the initial atom
    
    radial_wrapper.call_vint(handler_initial.rad_grid, rv_modified_initial)
    radial_wrapper.call_setrgrid(handler_initial.rad_grid)
    for i_s in range(len(handler_initial.dhfs_config.electron_config)):
        print(f"Shell {i_s:d}")
        n = handler_initial.dhfs_config.n_values[i_s]
        l = handler_initial.dhfs_config.l_values[i_s]
        j = 0.5*handler_initial.dhfs_config.jj_values[i_s]
        k = int((l-j)*(2*j+1))
        
        print(f"n = {n:d}, k = {k:d}")
        e_bound = 0.0
        try:
            e_bound = radial_wrapper.call_dbound(handler_initial.binding_energies[i_s],
                                    n,
                                    k,
                                    eps=1E-14)
        except RADIALError as e:
            print("Problem in obtaining bound state ", e)
        
        
        
    plt.show()
    
    
    

if __name__=="__main__":
    main()