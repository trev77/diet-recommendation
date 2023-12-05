import os
import json
import pickle
import requests
import pandas as pd

import logging
import wandb

class Meal:
    def __init__(self, id, food_codes, gram_portions, day_eaten, food_source):
        self.id = id
        self.food_codes = food_codes
        self.gram_portions = gram_portions
        self.day_eaten = day_eaten
        self.food_source = food_source

def configure_gpu(gpu_id):
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

def load_config(config_path):
    with open(config_path, 'r') as file:
        return json.load(file)
    
def to_pickle(data, filename):
    with open(filename, 'wb') as f:
        pickle.dump(data, f)
        
def setup_logger(caller_name, log_file, level=logging.INFO):
    """
    Create and configure a logger for the specified caller. 
    Each call can specify a new caller that will log to that file.
    """
    # Ensure the directory for the log file exists
    os.makedirs(os.path.dirname(log_file), exist_ok=True)

    # Create a formatter for logging
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')

    # Create a file handler for the specified log file
    file_handler = logging.FileHandler(log_file, mode='w')
    file_handler.setFormatter(formatter)

    # Create a logger with the name of the caller
    logger = logging.getLogger(caller_name)
    logger.setLevel(level)
    logger.addHandler(file_handler)

    # Avoid duplicate logging if logger already exists
    logger.propagate = False

    return logger
    
def setup_wandb(config, project_name, group_name, run_name, tags, job_type):
    wandb.init(
        project=project_name, 
        group=group_name, 
        name=run_name,
        tags=tags, 
        config=config, 
        job_type=job_type
    )
    
def download_file(url, local_filepath, log):
    try:
        with requests.get(url, stream=True) as r:
            r.raise_for_status()
            with open(local_filepath, 'wb') as f:
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)
    except Exception as e:
        log.error(f"Error downloading file {url}: {e}")      
           
def fetch_data(file_url, cache_dir, log):
    try:
        filename = os.path.basename(file_url)
        local_filepath = os.path.join(cache_dir, filename)

        if not os.path.exists(local_filepath):
            log.info(f"Downloading {filename}...")
            download_file(file_url, local_filepath, log)
        else:
            log.info(f"Using cached file: {filename}")

        if local_filepath.endswith(".XPT"):
            data = pd.read_sas(local_filepath)
        elif local_filepath.endswith(".xlsx"):
            data = pd.read_excel(local_filepath, engine='openpyxl', skiprows=1)
        else:
            raise ValueError(f"Unsupported file format for {local_filepath}")
        
        return data
    
    except Exception as e:
        log.error(f"Error fetching data from {file_url}: {e}")
        return pd.DataFrame()
