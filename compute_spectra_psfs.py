#!/usr/bin/env python
import os
import yaml
from argparse import ArgumentParser

def main(argv=None):
    '''Command line arguments'''
    parser = ArgumentParser(description="Compute spectra and PSFs for 2nubb decay")
    parser.add_argument("config_file")
    
    args = parser.parse_args()

    with open(args.config_file, 'r') as f:
        run_config = yaml.safe_load(f)
    
    print(run_config)

if __name__=="__main__":
    main()