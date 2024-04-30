import os
import re
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
    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')

    file_handler = logging.FileHandler(log_file, mode='w')
    file_handler.setFormatter(formatter)

    logger = logging.getLogger(caller_name)
    logger.setLevel(level)
    logger.addHandler(file_handler)
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
    
def extract_year_from_filename(filename):
    match = re.search(r'(\d{4})-', filename)
    return int(match.group(1)) if match else None
    
def download_file(url, local_filepath, log):
    try:
        with requests.get(url, stream=True) as r:
            r.raise_for_status()
            with open(local_filepath, 'wb') as f:
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)
    except Exception as e:
        if log is not None:
            log.error(f"Error downloading file {url}: {e}")      
           
def fetch_data(file_url, cache_dir, log):
    try:
        filename = os.path.basename(file_url)
        local_filepath = os.path.join(cache_dir, filename)
        
        if not os.path.exists(local_filepath):
            if log is not None:
                log.info(f"Downloading {filename}...")
            download_file(file_url, local_filepath, log)
        else:
            if log is not None:
                log.info(f"Using cached file: {filename}")
                
        if local_filepath.endswith(".XPT"):
            data = pd.read_sas(local_filepath)
        elif local_filepath.endswith(".xlsx"):
            data = pd.read_excel(local_filepath, engine='openpyxl', skiprows=1)
        else:
            raise ValueError(f"Unsupported file format for {local_filepath}")
        
        return data
    
    except Exception as e:
        if log is not None:
            log.error(f"Error fetching data from {file_url}: {e}")
        return pd.DataFrame()