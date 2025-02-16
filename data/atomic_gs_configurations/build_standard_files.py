import numpy as np
import yaml
import os
from ctypes import cdll
import re

shell_l = {"s":0, "p":1, "d":2, "f":3, "g":4, "h":5, "i":6}

def read_all_configs():
    try:
        _dir_name = os.path.dirname(__file__)
        input_file = open(os.path.join(_dir_name,"configs_nist.dat"), 'r')
    except FileNotFoundError as e:
        print(f"Error: Unable to find configs_nist.dat: {e}")
        
    lines = input_file.readlines()
    
    configs = {}
    for line in lines:
        if line[0] == "#":
            continue
        
        tokens = re.split(r'\s{2,}', line.strip())
        config_current = ""
        symbol = ""
        for tok in tokens[2].split():
            if tok[0] == "[":
                config_current = config_current+configs[tok[1:-1]]
            else:
                config_current = config_current+" "+tok
        
        configs[tokens[1]] = config_current.strip()
    return configs
    
def build_dirac_config(schrod_config:str):
    tokens = schrod_config.split()
    
    diract_config = []
    for tok in tokens:
        n = int(tok[0])
        l = shell_l[tok[1]]
        n_placed = int(tok[3:])
        
        for jj in [-1,1]:
            j2 = 2*l+jj
            if j2 < 0:
                continue
                
            n_max_subshell = j2+1
            n_placed_subshell = min(n_placed, n_max_subshell)
            
            diract_config.append([n,l,j2, n_placed_subshell])
            n_placed = n_placed-n_placed_subshell
            if (n_placed < 1):
                break
    
    return diract_config
    
def write_yaml(i_z:int, symbol:str, dirac_config:list[list[int]]):
    try:
        _dir_name = os.path.dirname(__file__)
        output_file = open(os.path.join(_dir_name,f"ground_state_config_Z{i_z:03d}.yaml"), 'w')
    except FileNotFoundError as e:
        print(f"Error: Unable to open output yaml file: {e}")
        
    text = f"""
---
Z: {i_z}
Name: {symbol}
configuration:
"""
    for shell in dirac_config:
        text = text + f"  - [{shell[0]},{shell[1]},{shell[2]},{shell[3]}]\n" 
    
    output_file.write(text)
    output_file.close()   
    
def build_all_files():
    schrod_configs = read_all_configs()
    
    i_z = 1
    for key in schrod_configs:
        dirac_config = build_dirac_config(schrod_configs[key])
        
        write_yaml(i_z, key, dirac_config)
        i_z = i_z + 1
    
if __name__ == "__main__":
    build_all_files()