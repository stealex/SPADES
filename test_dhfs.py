import numpy as np
import os
import yaml

from wrappers import dhfs_wrapper
from handlers.wavefunction_handlers import dhfs_handler
from utils import ph

import matplotlib.pyplot as plt

def test_dhfs():
    input_config_file = "data/ground_state_config_Z029.yaml"
    handler = dhfs_handler({"name": "29Cu", "weight": -1., "electron_config": input_config_file}, "Cu")
    handler.print()
    handler.run_dhfs(100)
    
    handler.retrieve_dhfs_results()
    
    fig, ax = plt.subplots(figsize=(8,6))
    ax.plot(handler.rad_grid, handler.rv_nuc)
    ax.plot(handler.rad_grid, -handler.rv_el)
    ax.plot(handler.rad_grid, handler.rv_ex)
    ax.plot(handler.rad_grid, (handler.rv_nuc+handler.rv_el+handler.rv_ex))
    ax.set_xscale('log')
    ax.set_xlim(1E-6, 1E1)
    
    # build the radial density
    density = np.zeros_like(handler.rad_grid)
    for i_shell in range(len(handler.dhfs_config.electron_config)):
        p = handler.p_grid[i_shell]
        q = handler.q_grid[i_shell]
        
        density_shell = (p*p)+(q*q)
        
        density = density + density_shell*handler.dhfs_config.electron_config[i_shell, 3]
        
    
    fig, ax = plt.subplots(1,2, figsize=(16,6))
    ax[0].plot(handler.rad_grid,density)
    ax[0].set_xscale('log')
    ax[0].set_xlim(2E-4, 1E1)
    
    ax[1].plot(handler.rad_grid,density/(4.0*np.pi*((handler.rad_grid)**2.0)))
    ax[1].set_xlim(0., 3.)
    ax[1].set_yscale('log')
    ax[1].set_ylim(1E-3,1E3)
    
    plt.show()
    
    
if __name__ == "__main__":
    test_dhfs()