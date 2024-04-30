import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from collections import defaultdict

from scipy.interpolate import interp1d
from utils.model_utils import aggregate_crossval_metrics, aggregate_crossval_stats_to_list

def aggregate_test_metrics(test_metrics):
    agg_metrics = defaultdict(list)
    for data in test_metrics:
        for key, value in data.items():
            agg_metrics[key].append(value)
    return agg_metrics

def get_meal_coverage(meals, base_value):
    return [x / base_value for x in meals]

def find_ideal_meals(foods_per_sparse, fold_means_r_squared, meals_per_sparse_normalized):
    line1_interp = interp1d(foods_per_sparse, fold_means_r_squared, kind='linear')
    line2_interp = interp1d(foods_per_sparse, meals_per_sparse_normalized, kind='linear')
    x_fine = np.linspace(min(foods_per_sparse), max(foods_per_sparse), 1000)
    differences = np.abs(line1_interp(x_fine) - line2_interp(x_fine))
    min_index = np.argmin(differences)
    ideal_num_meals = x_fine[min_index]
    closest_index = np.argmin(np.abs(np.array(foods_per_sparse) - ideal_num_meals))
    
    return ideal_num_meals, closest_index

def plot_sparsity_graphs(meal_type, plot_data, sparsity_info):
    fig = plt.figure(figsize=(15, 6))
    gs = gridspec.GridSpec(2, 2, width_ratios=[2, 1], height_ratios=[1, 1])
    fig.suptitle(f'{meal_type.capitalize()} Meal Portion Model Performance Across Sparsity Levels', fontsize=16)

    ax1 = plt.subplot(gs[:, 0])
    ax2 = ax1.twinx()
    ax1.plot(plot_data['foods_per_sparse'], plot_data['fold_means_r_squared'], 'bo-')
    ax2.plot(plot_data['foods_per_sparse'], plot_data['meals_per_sparse_normalized'], 'go-')
    ax1.fill_between(plot_data['foods_per_sparse'], 
                     [m - s for m, s in zip(plot_data['fold_means_r_squared'], plot_data['fold_stds_r_squared'])], 
                     [m + s for m, s in zip(plot_data['fold_means_r_squared'], plot_data['fold_stds_r_squared'])], 
                     color='gray', alpha=0.3)
    
    ax1.set_xlabel('Number of Foods')
    ax1.set_ylabel('Predicted vs. actual portion size (R², 5-fold CV)', color='b')
    ax2.set_ylabel('Meal Coverage (%)', color='g')
    ax1.tick_params(axis='y', colors='b')
    ax2.tick_params(axis='y', colors='g')
    
    optimal_index = sparsity_info['closest_index']
    optimal_foods_per_sparse = plot_data['foods_per_sparse'][optimal_index]
    optimal_r_squared = plot_data['fold_means_r_squared'][optimal_index]
    ax1.plot(optimal_foods_per_sparse, optimal_r_squared + 0.05 * optimal_r_squared, 'r*', markersize=10)

    y_low = 0.3 if meal_type == 'breakfast' else 0.1 # for publication results
    ax1.set_ylim([y_low, 1.1])
    ax2.set_ylim([y_low, 1.1])

    ax3 = plt.subplot(gs[0, 1])
    ax3.plot(sparsity_info['sparsity_removed_levels'], plot_data['meals_per_sparse'], '-ko')
    ax3.plot(sparsity_info['corresponding_sparsity_level'], 
             plot_data['meals_per_sparse'][sparsity_info['closest_index']] + 0.15 * plot_data['meals_per_sparse'][sparsity_info['closest_index']], 
             'r*', markersize=10)
    ax3.set_ylabel('Number of Meals')

    ax4 = plt.subplot(gs[1, 1], sharex=ax3)
    ax4.plot(sparsity_info['sparsity_removed_levels'], plot_data['test_r_squared'], '-ko')
    ax4.plot(sparsity_info['corresponding_sparsity_level'], 
             plot_data['test_r_squared'][sparsity_info['closest_index']] - 0.045 * plot_data['test_r_squared'][sparsity_info['closest_index']], 
             'r*', markersize=10)
    ax4.set_xlabel('Sparsity Removed (%)')
    ax4.set_ylabel('R-squared')
    plt.subplots_adjust(wspace=0.3, hspace=0.3)

    return fig

def sparsity_reduction_visualization(res_dir, foods_per_sparse, meals_per_sparse, fold_metrics, test_metrics, log):
    cv_stats = [aggregate_crossval_metrics(fold_metrics[i]) for i in range(len(fold_metrics))]
    cv_stats_list = aggregate_crossval_stats_to_list(cv_stats)

    fold_means_r_squared = cv_stats_list['f1']['mean']
    fold_stds_r_squared = cv_stats_list['f1']['std']
    test_stats_list = aggregate_test_metrics(test_metrics)
    test_r_squared = test_stats_list['f1']
    sparsity_removed_levels = [i for i in range(0, 100, 5)]
    meals_per_sparse_normalized = get_meal_coverage(meals_per_sparse, meals_per_sparse[0])
    meal_type = res_dir.split("/")[2]

    ideal_num_meals, closest_index = find_ideal_meals(foods_per_sparse, fold_means_r_squared, meals_per_sparse_normalized)
    corresponding_sparsity_level = sparsity_removed_levels[closest_index]
    
    plot_data = {
        'foods_per_sparse': foods_per_sparse,
        'meals_per_sparse': meals_per_sparse,
        'fold_means_r_squared': fold_means_r_squared,
        'fold_stds_r_squared': fold_stds_r_squared,
        'meals_per_sparse_normalized': meals_per_sparse_normalized,
        'test_r_squared': test_r_squared
    }
    sparsity_info = {
        'corresponding_sparsity_level': corresponding_sparsity_level,
        'closest_index': closest_index,
        'sparsity_removed_levels': sparsity_removed_levels
    }
    fig = plot_sparsity_graphs(meal_type, plot_data, sparsity_info)

    savefile = os.path.join(res_dir, 'sparsity_visualization.svg')
    plt.savefig(savefile, format='svg', dpi=1000)
    log.info(f"{meal_type.capitalize()} intersection plot saved to {savefile}")

    plt.close(fig)
    return corresponding_sparsity_level