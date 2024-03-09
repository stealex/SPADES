import yaml
import periodictable
import re
import os
import numpy as np
        
class run_input:
    def __init__(self, config_file_name:str) -> None:
        config = yaml.safe_load(config_file_name)
        
        self.task_name = config["task"]
        self.process_name = config["process"]
        self.output_dir = config["output_directory"]
        
        self.initial_atom = {"name": config["initial_atom"]["name"],
                             "weight": config["initial_atom"]["weight"],
                             "electron_config": config["initial_atom"]["electron_config"]}
        
        self.final_atom = {"electron_config": config["final_atom"]["electron_config"]}
        
        