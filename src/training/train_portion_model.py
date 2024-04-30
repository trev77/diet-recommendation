import os
import sys
import pandas as pd
from sklearn.model_selection import train_test_split

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.helpers import *
from utils.model_utils import *
from utils.models import PortionModel
from evaluation.evaluate_sparsity_filtration import *
    
def train_portion_model(data, model, meal_type, wb_group_name, log):
    X = (data > 0).astype(int)
    y = data.copy()
    config = load_config('config.json')
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=42)
    wb_project_name = f'{meal_type}_portion_model'
    fold_macro_metrics, fold_micro_metrics, best_model = cross_validation(
        X=X_train, 
        y=y_train, 
        model_configs=model, # messy, needs fixing
        meal_type=meal_type, 
        model_type='portion', 
        config=config, 
        project_name=wb_project_name, 
        wb_group_name=wb_group_name,
        log=log
    )
    test_set = {'X_test': X_test, 'y_test': y_test}
    to_pickle(test_set, f'{model.results_save_path}/test_set.pkl')
    results = log_metrics(X_test, y_test, best_model, fold_macro_metrics, fold_micro_metrics, log)
    
    return results

def remove_columns_and_rows(data, percentage_to_remove):
    num_columns_to_remove = int(data.shape[1] * percentage_to_remove / 100)
    sorted_columns = (data == 0).sum().sort_values(ascending=False).index[:num_columns_to_remove]
    reduced_data = data.drop(columns=sorted_columns)
    rows_to_remove = data[sorted_columns].gt(0).any(axis=1)
    reduced_data = reduced_data.drop(index=data[rows_to_remove].index)
    reduced_data = reduced_data[(reduced_data != 0).any(axis=1)]
    
    return reduced_data

def construct_save_path(meal_type, model_type, save_type, config):
    if model_type == 'portion_model':
        logdir = '_'.join(str(size) for size in config[model_type]["layer_sizes"])
    elif model_type == 'presence_model':
        logdir = f'{config[model_type]["learning_rate"]}_{config[model_type]["latent_dim"]}'
    logdir += f'_{config[model_type]["epochs"]}_{config[model_type]["batch_size"]}'
    
    if save_type == 'model':
        path = f'models/{model_type}/{meal_type}/{logdir}/'
    elif save_type == 'results':
        path = f'results/{model_type}/{meal_type}/{logdir}/'
    elif save_type == 'logging':
        path = f'logs/{model_type}/{meal_type}/{logdir}/'
    if not os.path.exists(path): os.makedirs(path)
    
    return path