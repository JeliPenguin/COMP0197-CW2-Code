import torch
import numpy as np
import json 
import os 
import torch
import argparse

# from src.mae_code.model import MAE
from src.mae_code.mae_arch import MAE



def save_config(args):
    """
    Saves the training configuration in a JSON file in a specified directory.
    """
    file_name = f'config_{args.run_id}.json'
    # Ensure the directory exists
    directory=  args.checkpoint_dir
    os.makedirs(directory, exist_ok=True)
    
    # Full path to the configuration file
    file_path = os.path.join(directory, file_name)

    with open(file_path, 'w') as f:
        # Convert args to dictionary (if it's an argparse.Namespace)
        args_dict = vars(args)
        # we've saved  two tensors to args. serialize and unserialize in config to model
        args_dict['mean_pixels'] = args.mean_pixels.tolist()
        args_dict['std_pixels'] = args.std_pixels.tolist()

        # handle non serialiable objects specifically
        if 'device' in args_dict:
            args_dict['device'] = str(args.device)  # Convert device to string

        json.dump(args_dict, f, indent=4)
        
def load_config(filename='config.json'):
    """
    Loads the configuration from a JSON file.
    """
    with open(filename, 'r') as f:
        config = json.load(f)
        return config
    

def load_model(model_path):
    config = load_config(os.path.join(model_path,"config.json"))
    args = argparse.Namespace(**config)
    
    #unserialize
    args.mean_pixels = torch.tensor(config['mean_pixels'])
    args.std_pixels = torch.tensor(config['std_pixels'])

    model = MAE(args)  
    model.load_state_dict(torch.load(os.path.join(model_path,"model.pth")))
    return model, args
