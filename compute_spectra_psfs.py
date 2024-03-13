#!/usr/bin/env python
import os
import yaml
from argparse import ArgumentParser
from configs.wavefunctions_config import run_config
from wavefunctions_computation.wavefunctions import wavefunctions_handler
import matplotlib.pyplot as plt
import numpy as np

   
def main(argv=None):
    '''Command line arguments'''
    parser = ArgumentParser(description="Compute spectra and PSFs for 2nubb decay")
    parser.add_argument("config_file",
                        help="path to yaml configuration file")
    
    args = parser.parse_args()

    with open(args.config_file, 'r') as f:
        run_conf = yaml.safe_load(f)
    
    input_config = run_config(run_conf)
    
    wf_handler_initial = wavefunctions_handler(input_config.initial_atom, input_config.bound_config)
    wf_handler_initial.find_all_wavefunctions()
    
    wf_handler_final = wavefunctions_handler(input_config.final_atom, input_config.bound_config, input_config.scattering_config)
    wf_handler_final.find_all_wavefunctions()
    
if __name__=="__main__":
    main()