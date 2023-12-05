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

def train_models_by_sparsity(data, meal_type, preprocess_log):
    outdir = f'data/processed/meals/dataframes/{meal_type}'

    #if not os.path.exists(f'{outdir}_sparsity_filtered.pkl'):
    foods_per_sparse, meals_per_sparse = [], []
    fold_macro_results, fold_micro_results = [], []
    test_macro_results, test_micro_results = [], []
    
    config = load_config('config.json')
    for percentage in [i  for i in range(0, 100, 5)]:
        wb_group_name = f"sparsity_{percentage}"

        respath = construct_save_path(meal_type, 'portion_model', 'results', config)
        respath += f'sparsity_{percentage}/'
        os.makedirs(respath, exist_ok=True)
        logpath = construct_save_path(meal_type, 'portion_model', 'logging', config)
        logpath += f'sparsity_{percentage}'
        sparsity_logfile = f'{logpath}_training.log'
        sparsity_model_train_log = setup_logger(f'sparsity_{percentage}', sparsity_logfile)

        reduced_data = remove_columns_and_rows(data, percentage)
        meals_per_sparse.append(reduced_data.shape[0])
        foods_per_sparse.append(reduced_data.shape[1])
        sparsity_model_train_log.info(f'Dataset reduced to {reduced_data.shape[0]} meals and {reduced_data.shape[1]} foods')

        log_model_params(config, 'portion_model', sparsity_model_train_log)
        sparsity_filtration_path = f'data/processed/meals/dataframes/sparsity_filtration/{meal_type}/'
        os.makedirs(sparsity_filtration_path, exist_ok=True)
        to_pickle(reduced_data, sparsity_filtration_path + f'{meal_type}_sparse_{int(percentage)}.pkl')
        
        preprocess_log.info(f"Training model at {int(percentage)}% sparse columns removed...")
        if not os.path.exists(f'{respath}result.pkl'):
            model = PortionModel(reduced_data.shape[1], meal_type, 'config.json')
            model.results_save_path += f'sparsity_{int(percentage)}/'
            model.model_save_path += f'sparsity_{int(percentage)}/'
            fold_macro_metrics, fold_micro_metrics, test_metrics = train_portion_model(reduced_data, model, meal_type, wb_group_name, sparsity_model_train_log)
            result = fold_macro_metrics, fold_micro_metrics, test_metrics
            to_pickle(result, f'{respath}result.pkl')
        else:
            sparsity_model_train_log.info(f"{respath}result.pkl already exists, using cached data")            
            fold_macro_metrics, fold_micro_metrics, test_metrics = pd.read_pickle(f'{respath}result.pkl')
            
        fold_macro_results.append(fold_macro_metrics)
        fold_micro_results.append(fold_micro_metrics)
        test_macro_results.append(test_metrics[0])
        test_micro_results.append(test_metrics[1])

    results_save_path = construct_save_path(meal_type, 'portion_model', 'results', config) 

    preprocess_log.info(f"Visualizing and extracting ideal sparsity level...")
    ideal_sparsity_percentage = sparsity_reduction_visualization(results_save_path, foods_per_sparse, meals_per_sparse, fold_micro_results, test_micro_results, preprocess_log)
    preprocessed_data = remove_columns_and_rows(data, ideal_sparsity_percentage)
    preprocess_log.info(f"Sparsity filtered dataset has {preprocessed_data.shape[0]} meals and {preprocessed_data.shape[1]} foods")
    to_pickle(preprocessed_data, f'{outdir}_sparsity_filtered.pkl')
    #else:
    #    sparsity_filtration_log.info(f'{outdir}_sparsity_filtered.pkl already exists, using cached data')