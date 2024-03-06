import numpy as np
import os
import yaml

from wrappers import dhfs_wrapper

def test_dhfs():
    input_config_file = "data/ground_state_config_Z020.yaml"
    with open(input_config_file, 'r') as f:
        electron_config = yaml.safe_load(f)
        
    print(electron_config)
    shell_config = np.array(electron_config["configuration"])
    print(shell_config)
    n_values = shell_config[:,0].astype(int)
    l_values = shell_config[:,1].astype(int)
    jj_values = shell_config[:,2].astype(int)
    occ_values = shell_config[:,3].astype(float)
    
    dhfs_wrapper.call_configuration_input(n_values, l_values, jj_values, occ_values, 20)
    dhfs_wrapper.call_set_parameters(40.0, 100., 1000)
    dhfs_wrapper.call_dhfs_main(1.5)
    
if __name__ == "__main__":
    test_dhfs()