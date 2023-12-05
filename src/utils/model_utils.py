import os
import sys
import statistics
from collections import defaultdict
from sklearn.model_selection import KFold

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.helpers import *
from utils.models import *

def format_metrics(metrics, is_test=False):
    formatted = []
    for metric, values in metrics.items():
        if is_test:
            formatted.append(f"{metric.capitalize()}: {values:>8.10f}")
        else:
            formatted.append(f"{metric.capitalize()}: {values['mean']:>8.10f} ± {values['std']:>8.10f}")
    return ', '.join(formatted)

def wandb_log_metrics(micro_results, macro_results, wb_group_name, fold, log):
    if 'param_sweep' in wb_group_name:
        wandb.log({"fold": fold, **micro_results})
    else:
        wandb.log({"sparsity_level": int(wb_group_name.split('_')[1]), "fold": fold, **micro_results})
    wandb.finish()

    log.info(f"Fold {fold} Metrics (Macro) - {format_metrics(macro_results, is_test=True)}")
    log.info(f"Fold {fold} Metrics (Micro) - {format_metrics(micro_results, is_test=True)}")

def log_model_params(config, model_type, log):
    if model_type == 'portion_model':
        log.info(f'Model parameters - {len(config[model_type]["layer_sizes"])} layers of size {config[model_type]["layer_sizes"]}')
    elif model_type == 'presence_model':
        log.info(f'Model parameters - {len(config[model_type]["encoder_sizes"])} encoder layers of size {config[model_type]["encoder_sizes"]}')
        log.info(f'Model parameters - {len(config[model_type]["decoder_sizes"])} decoder layers of size {config[model_type]["decoder_sizes"]}')
    log.info(f'Model parameters - Dropout rate: {config[model_type]["dropout_rate"]}')
    log.info(f'Model parameters - Epochs: {config[model_type]["epochs"]}')
    log.info(f'Model parameters - Batch size: {config[model_type]["batch_size"]}')

def log_metrics(X, y, best_model, fold_macro_metrics, fold_micro_metrics, log):
    cv_macro_stats = aggregate_crossval_metrics(fold_macro_metrics)
    cv_micro_stats = aggregate_crossval_metrics(fold_micro_metrics)
    test_micro_metrics, test_macro_metrics = best_model.evaluate(X, y)
    
    log.info("=" * 61 + " Results Summary " + "=" * 63)
    log.info(f"CrossVal Metrics (Macro) - {format_metrics(cv_macro_stats)}")
    log.info(f"Testing  Metrics (Macro) - {format_metrics(test_macro_metrics, is_test=True)}")
    log.info(f"CrossVal Metrics (Micro) - {format_metrics(cv_micro_stats)}")
    log.info(f"Testing  Metrics (Micro) - {format_metrics(test_micro_metrics, is_test=True)}")

    return fold_macro_metrics, fold_micro_metrics, (test_macro_metrics, test_micro_metrics)

def aggregate_crossval_metrics(fold_metrics):
    values_by_key = defaultdict(list)
    for d in fold_metrics:
        for key, value in d.items():
            values_by_key[key].append(value)
    stats_by_key = {
        key: {'mean': statistics.mean(values),'std': statistics.stdev(values)} 
        for key, values in values_by_key.items()
    }
    return stats_by_key

def aggregate_crossval_stats_to_list(results_by_sparsity_level):
    aggregated_stats = defaultdict(lambda: {'mean': [], 'std': []})
    for stats_dict in results_by_sparsity_level:
        for key, stats in stats_dict.items():
            aggregated_stats[key]['mean'].append(stats['mean'])
            aggregated_stats[key]['std'].append(stats['std'])
    aggregated_stats = dict(aggregated_stats)

    return aggregated_stats
        
def instantiate_model(meal_type, model_type, input_size, logpath):
    if model_type == 'portion':
        model = PortionModel(input_size, meal_type, 'config.json')
        model.results_save_path = logpath
    elif model_type == 'presence':
        model = PresenceModel(input_size, meal_type, 'config.json')
    return model

def cross_validation(X, y, model_configs, meal_type, model_type, config, project_name, wb_group_name, log, num_folds=5):
    kfold = KFold(n_splits=num_folds, shuffle=True)
    fold_macro_metrics, fold_micro_metrics = [], []
    best_model = None
    best_score = 0
    
    for fold, (train_index, val_index) in enumerate(kfold.split(X)):
        wb_run_name = f"fold_{fold}"
        setup_wandb(
            config, 
            project_name, wb_group_name, wb_run_name, 
            [meal_type, wb_group_name, wb_run_name], 
            f'{num_folds}-fold cross-validation'
        )
        log.info(f"="*66 + f" Fold {fold + 1} " + f"="*67)
        X_fold_train, X_fold_val = X.iloc[train_index], X.iloc[val_index]
        y_fold_train, y_fold_val = y.iloc[train_index], y.iloc[val_index]
        
        model = instantiate_model(meal_type, model_type, X_fold_train.shape[1], model_configs.results_save_path)
        model.train(X_fold_train, y_fold_train, X_fold_val, y_fold_val, log)
        micro_results, macro_results = model.evaluate(X_fold_val, y_fold_val)

        if micro_results[model.config[f'{model_type}_model']['eval_metric']] > best_score:
            best_score = micro_results[model.config[f'{model_type}_model']['eval_metric']]
            best_model = model
        wandb_log_metrics(micro_results, macro_results, wb_group_name, fold, log)

        fold_macro_metrics.append(macro_results)
        fold_micro_metrics.append(micro_results)
        fold_data = {'train_index': train_index, 'val_index': val_index}
        to_pickle(fold_data, model.results_save_path + f'fold_{fold+1}_indices.pkl')

        model.model_save_path = model_configs.model_save_path # messy, need to fix this later
        os.makedirs(model.model_save_path, exist_ok=True)
        model.model_save_path += f'fold_{fold+1}_'
        model.results_save_path += f'fold_{fold+1}_'

        model.save_model(log)

    return fold_macro_metrics, fold_micro_metrics, best_model
