#!/usr/bin/env python
import os
import yaml
from argparse import ArgumentParser
from utils.run_handler import run_input
from utils.dhfs_handler import dhfs_handler
import numpy as np
from utils import physics_constants

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
    
    # bound states of final atom
    handler_final = dhfs_handler(input_config.final_atom, "final_atom")
    handler_final.run_dhfs(input_config.bound_config.max_r,
                           input_config.bound_config.n_radial_points, 
                           input_config.bound_config.radius_unit)
    handler_final.retrieve_dhfs_results()
    
    # build modified potentials
    rv_modified_initial = handler_initial.build_modified_potential()
    rv_modified_final = handler_final.build_modified_potential()
    
    fig, ax = plt.subplots(figsize=(8,6))
    ax.plot(handler_initial.rad_grid, rv_modified_initial)
    ax.plot(handler_initial.rad_grid, handler_initial.rv_nuc+
                                      handler_initial.rv_el+
                                      handler_initial.rv_ex)
    ax.set_xscale('log')
    
    # solve bound states in the initial atom
    
    plt.show()
    
    
    

if __name__=="__main__":
    main()