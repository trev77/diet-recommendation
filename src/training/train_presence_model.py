import os
import sys
import pandas as pd
from sklearn.model_selection import train_test_split

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.helpers import *
from utils.model_utils import *

def train_presence_model(data, meal_type, wb_group_name, presence_model_log, config):
    log_model_params(config, 'presence_model', presence_model_log)

    bin_data = data.applymap(lambda x: 1 if x > 0 else x)
    X_train, X_test = train_test_split(bin_data, test_size=0.15, random_state=42)
    
    wb_project_name = f'{meal_type}_presence_model'
    fold_macro_metrics, fold_micro_metrics, best_model = cross_validation(
        X=X_train, 
        y=X_train, 
        logpath='', 
        meal_type=meal_type, 
        model_type='presence', 
        config=config, 
        project_name=wb_project_name, 
        wb_group_name=wb_group_name,
        log=presence_model_log
    )
    test_set = {'X_test': X_test}
    to_pickle(test_set, '/'.join(best_model.results_save_path.split('/')[0:-1]) + f'/test_set.pkl')
    results = log_metrics(X_test, X_test, best_model, fold_macro_metrics, fold_micro_metrics, presence_model_log)
    
    return results

def main():
    meal_type = sys.argv[1]    
    configure_gpu(-1)
    config = load_config('config.json')

    logdir = f'{config["presence_model"]["learning_rate"]}_{config["presence_model"]["latent_dim"]}'
    logdir += f'_{config["presence_model"]["epochs"]}_{config["presence_model"]["batch_size"]}'
    presence_model_log = setup_logger('presence_model_log', f'logs/{meal_type}/presence_model/{logdir}/presence_model_training.log')
    data = pd.read_pickle(f'data/processed/meals/dataframes/{meal_type}_sparsity_filtered.pkl')
    
    presence_model_log.info(f"Training presence model...")
    wb_group_name = 'first_attempt_param_sweep'
    train_presence_model(data, meal_type, wb_group_name, presence_model_log, config)

if __name__ == '__main__':
    main()
