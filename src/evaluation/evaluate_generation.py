import os
import sys
import numpy as np
import pandas as pd
from collections import Counter
from joblib import Parallel, delayed

import matplotlib.pyplot as plt
from scipy.stats import entropy
from scipy.spatial.distance import cosine

from pprint import pprint
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.helpers import *

MACRONUTRIENTS = {
    'Carbohydrate (g)',
    'Total Fat (g)',
    'Protein (g)',
    'Fiber (total dietary) (g)',
    'Monounsaturated fats (g)',
    'Polyunsaturated fats (g)',
    'Saturated fats (g)',
    'Stearic Acid (18:0) (g)',
    'Energy (kcal)'  # Energy can also be considered as part of macronutrients due to its direct relation with macronutrient intake
}

MICRONUTRIENTS = {
    'Calcium (mg)',
    'Cholesterol (mg)',
    'Choline (mg)',
    'Copper (mg)',
    'DHA (22:6 n-3) (g)',
    'EPA (20:5 n-3) (g)',
    'Folate (mg_DFE)',
    'Iron (mg)',
    'Magnesium (mg)',
    'Niacin (mg)',
    'Phosphorus (mg)',
    'Potassium (mg)',
    'Riboflavin (mg)',
    'Sodium (mg)',
    'Thiamin (mg)',
    'Vitamin A (mg_RAE)',
    'Vitamin B-12 (mcg)',
    'Vitamin B-6 (mg)',
    'Vitamin C (mg)',
    'Vitamin D (IU)',
    'Vitamin E (alpha-tocopherol) (mg)',
    'Vitamin K (mg)',
    'Zinc (mg)'
}

NUTRIENT_NAME_MAPPING = {
    'Fiber, total dietary (g)': 'Fiber (total dietary) (g)',
    'Fatty acids, total saturated (g)': 'Saturated fats (g)',
    'Fatty acids, total monounsaturated (g)': 'Monounsaturated fats (g)',
    'Fatty acids, total polyunsaturated (g)': 'Polyunsaturated fats (g)',
    '20:5 n-3 (g)': 'EPA (20:5 n-3) (g)',
    '22:6 n-3 (g)': 'DHA (22:6 n-3) (g)',
    '18:0 (g)': 'Stearic Acid (18:0) (g)',
    'Choline, total (mg)': 'Choline (mg)',
    'Vitamin A, RAE (mcg_RAE)': 'Vitamin A (mg_RAE)', # divide mcg_RAE by 1000 to get mg_RAE
    'Vitamin D (D2 + D3) (mcg)': 'Vitamin D (IU)', # divide mcg by 0.025 to get IU
    'Vitamin K (phylloquinone) (mcg)': 'Vitamin K (mg)', # divide mcg by 1000 to get mg 
    'Folate, total (mcg)': 'Folate (mg_DFE)'  # divide mcg by 0.6 to get mcg DFE 
}
    
"""
def aggregate_nutrition_per_day(bootstrap_sample_dict, num_days):
    aggregated_nutrition = pd.DataFrame()
    meal_num = 0
    for _ in range(1, num_days + 1):
        daily_nutrition = pd.Series(dtype=float)
        for meal_type in bootstrap_sample_dict.keys():
            meal_df = bootstrap_sample_dict[meal_type]
            meal_nutrition = calculate_meal_nutrition_from_df(meal_df.iloc[meal_num])
            daily_nutrition = daily_nutrition.add(meal_nutrition, fill_value=0)
        daily_nutrition_row = pd.DataFrame([daily_nutrition])
        aggregated_nutrition = pd.concat([aggregated_nutrition, daily_nutrition_row], ignore_index=True)
        meal_num += 1
    return aggregated_nutrition

def standardize_nutrient_fields(aggregated_nutrition_per_day, calorie_level='2000 cal'):
    nutrient_targets = pd.read_csv('data/raw/food_patterns/healthy_US_style_food_pattern.csv', index_col=0)
    if calorie_level not in nutrient_targets.columns:
        raise ValueError(f"Calorie level {calorie_level} not found in nutrient targets.")
    target_nutrients = nutrient_targets[calorie_level]
    aggregated_nutrition_per_day = nutrient_mapping_and_conversions(aggregated_nutrition_per_day)
    return target_nutrients, aggregated_nutrition_per_day

def calculate_nutrient_coverage(bootstrap_sample_dict, meal_specific_targets, nutrient_fields):
    coverage_scores = {meal_type: [] for meal_type in bootstrap_sample_dict} 

    for meal_type, meal_df in bootstrap_sample_dict.items():
        target_nutrients = meal_specific_targets[meal_type]
        
        for _, meal in tqdm(meal_df.iterrows(), total=meal_df.shape[0]):
            meal_nutrition = calculate_meal_nutrition_from_df(meal)
            meal_nutrition = nutrient_mapping_and_conversions(pd.DataFrame([meal_nutrition]))
            nutrient_coverage = {}

            for nutrient, rdi in target_nutrients.items():
                if nutrient in nutrient_fields:
                    nutrient_amount = meal_nutrition.get(nutrient, 0)
                    coverage_ratio = nutrient_amount / rdi if rdi > 0 else 0
                    coverage = 1 - abs(1 - coverage_ratio)
                    nutrient_coverage[nutrient] = coverage

            coverage_scores[meal_type].append(nutrient_coverage)  # Append to the list
    # Calculate average coverage scores for each meal type
    avg_coverage_scores = {meal_type: pd.DataFrame(scores).mean().to_dict() 
                           for meal_type, scores in coverage_scores.items()}
    return avg_coverage_scores

def calculate_nutrient_coverage(aggregated_nutrition_per_day, nutrient_fields):
    target_nutrients, aggregated_nutrition_per_day = standardize_nutrient_fields(aggregated_nutrition_per_day)
    daily_coverage_scores = []
    for _, daily_nutrition in aggregated_nutrition_per_day.iterrows():
        nutrient_coverage = {}
        for nutrient, rdi in target_nutrients.iteritems():
            if nutrient in nutrient_fields:
                nutrient_amount = daily_nutrition.get(nutrient, 0)
                coverage_ratio = nutrient_amount / rdi if rdi > 0 else 0
                coverage = 1 - abs(1 - coverage_ratio)
                nutrient_coverage[nutrient] = coverage
        daily_coverage_scores.append(nutrient_coverage)

    avg_coverage_scores = pd.DataFrame(daily_coverage_scores).mean().to_dict()
    overall_average_coverage = sum(avg_coverage_scores.values()) / len(avg_coverage_scores) if avg_coverage_scores else 0

    if 'Energy (kcal)' in nutrient_fields:
        label = 'Average Macronutrient Coverage'
    else:
        label = 'Average Micronutrient Coverage'
    plot_spider(avg_coverage_scores, label)
    
    return overall_average_coverage

def calculate_nutrient_density(aggregated_nutrition_per_day, nutrient_fields):
    target_nutrients, aggregated_nutrition_per_day = standardize_nutrient_fields(aggregated_nutrition_per_day)
    daily_nutrient_densities = {nutrient: [] for nutrient in nutrient_fields if nutrient != 'Energy (kcal)'}    
    for _, daily_nutrition in aggregated_nutrition_per_day.iterrows():
        total_calories = daily_nutrition['Energy (kcal)']
        
        for nutrient in nutrient_fields:
            if nutrient != 'Energy (kcal)':
                nutrient_amount = daily_nutrition.get(nutrient, 0)
                usda_target = target_nutrients.loc[nutrient]
                nutrient_density = (nutrient_amount / total_calories) / usda_target * 1000 if total_calories > 0 and usda_target > 0 else 0
                daily_nutrient_densities[nutrient].append(nutrient_density) 
    
    average_nutrient_densities = {nutrient: sum(densities) / len(densities) for nutrient, densities in daily_nutrient_densities.items()}
    overall_average_density = sum(average_nutrient_densities.values()) / len(average_nutrient_densities) if average_nutrient_densities else 0
    
    if 'Energy (kcal)' in nutrient_fields:
        title = 'Average Macronutrient Density'
    else:
        title = 'Average Micronutrient Density'
    plot_spider(average_nutrient_densities, title)

    return overall_average_density
    
def add_to_spider(ax, angles, values, color, label):
    values = values.tolist()
    values += values[:1]  # Complete the loop
    ax.plot(angles, values, color=color, linewidth=2, label=label)
    ax.fill(angles, values, color=color, alpha=0.25)

def plot_spider(nutrient_metrics, title='Nutrient Metrics'):
    nutrients = list(nutrient_metrics.keys())
    num_vars = len(nutrients)
    angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(8, 6), subplot_kw=dict(polar=True))

    metrics_series = pd.Series(nutrient_metrics, index=nutrients)
    add_to_spider(ax, angles, metrics_series, 'b', title)

    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    ax.set_thetagrids(np.degrees(angles[:-1]), nutrients, fontsize=10)

    for label, angle in zip(ax.get_xticklabels(), angles):
        label.set_horizontalalignment('center' if angle in [0, np.pi] else 'left' if 0 < angle < np.pi else 'right')
        label.set_rotation(angle - 90)

    plt.legend(loc='upper right', bbox_to_anchor=(0,0))
    plt.title(f'Nutritional Comparison: {title}')
    plt.savefig(f'tests/{title}.png', dpi=1000)
""" 

def nutrient_mapping_and_conversions(aggregated_nutrition_per_day):
    aggregated_nutrition_per_day.columns = aggregated_nutrition_per_day.columns.str.replace('\n', ' ')
    aggregated_nutrition_per_day.rename(columns=NUTRIENT_NAME_MAPPING, inplace=True)
    """
    if 'Vitamin A (IU)' in aggregated_nutrition_per_day.columns:
        aggregated_nutrition_per_day['Vitamin A (IU)'] = aggregated_nutrition_per_day['Vitamin A (IU)'] * 12  
    if 'Vitamin D (IU)' in aggregated_nutrition_per_day.columns:
        aggregated_nutrition_per_day['Vitamin D (IU)'] = aggregated_nutrition_per_day['Vitamin D (IU)'] * 40
    if 'Vitamin K (mg)' in aggregated_nutrition_per_day.columns:
        aggregated_nutrition_per_day['Vitamin K (mg)'] = aggregated_nutrition_per_day['Vitamin K (mg)'] / 1000
    if 'Folate (mg_DFE)' in aggregated_nutrition_per_day.columns:
        aggregated_nutrition_per_day['Folate (mg_DFE)'] = aggregated_nutrition_per_day['Folate (mg_DFE)'] / 0.6
    """
    return aggregated_nutrition_per_day

def calculate_average_nutrient_values(meal_dict):
    average_nutrients = {}
    for meal_type, meal_df in meal_dict.items():
        meal_nutrition = meal_df.apply(lambda row: calculate_meal_nutrition_from_df(row), axis=1)
        total_nutrients = meal_nutrition.sum()
        average_nutrients[meal_type] = total_nutrients / len(meal_df)
    return average_nutrients

def calculate_nutrient_proportions(real_meals_dict):    
    average_nutrient_values_file = 'data/processed/real_meals/dataframes/average_nutrient_values.pkl'
    if not os.path.exists(average_nutrient_values_file):
        real_avg = calculate_average_nutrient_values(real_meals_dict)
        to_pickle(real_avg, average_nutrient_values_file)
    else:
        real_avg = pd.read_pickle(average_nutrient_values_file)
    nutrient_proportions = {meal_type: real_avg[meal_type]/sum(real_avg.values()) for meal_type in real_avg.keys()}

    return nutrient_proportions

"""
def scale_rd_to_meal_proportions(nutrient_proportions, calorie_level="2000 cal"):
    nutrient_targets = pd.read_csv('data/raw/food_patterns/healthy_US_style_food_pattern.csv', index_col=0)
    target_nutrients = nutrient_targets[calorie_level]
    scaled_targets = {}
    for meal_type, proportions in nutrient_proportions.items():
        meal_specific_targets = {}
        for nutrient in target_nutrients.index:
            nutrient_prop = proportions.get(nutrient, 1)  # Default to 1 if not found
            meal_specific_targets[nutrient] = target_nutrients[nutrient] * nutrient_prop
        scaled_targets[meal_type] = meal_specific_targets

    return scaled_targets
"""

def load_nutrient_targets(calorie_level='2000 cal'):
    # Load the nutrient targets from the CSV file
    nutrient_targets_df = pd.read_csv('data/raw/food_patterns/healthy_US_style_food_pattern.csv', index_col=0)
    # Select the column corresponding to the specified calorie level
    nutrient_targets = nutrient_targets_df[calorie_level]
    return nutrient_targets

def calculate_meal_nutrition_from_df(meal_df, nutrient_data):
    meal_df = meal_df[meal_df > 0].reset_index()
    meal_df.columns = ['food_code', 'portion']
    meal_df['food_code'] = meal_df['food_code'].astype(int)
    
    # Ensure nutrient_data is indexed by 'Food code' for easy lookup
    relevant_nutrients = nutrient_data.loc[nutrient_data.index.intersection(meal_df['food_code'])]
    # Merge meal_df with relevant_nutrients to match portion sizes with nutrient content
    merged_data = meal_df.merge(relevant_nutrients, left_on='food_code', right_index=True)
    # Scale the nutrients from per 100g basis to per gram basis
    # Then, multiply by the portion size for each food item in the meal
    # This involves scaling nutrient values in all columns except 'food_code' and 'portion'
    nutrient_columns = [col for col in merged_data.columns if col not in ['food_code', 'portion']]
    scaled_nutrients = merged_data[nutrient_columns].div(100).mul(merged_data['portion'], axis=0)
    # Sum the scaled nutrient values across all food items to get the total nutrient content of the meal
    meal_nutrients = scaled_nutrients.sum()
    meal_nutrients.index = meal_nutrients.index.str.replace('\n', ' ', regex=True)
    meal_nutrients.rename(index=NUTRIENT_NAME_MAPPING, inplace=True)
    return meal_nutrients

def aggregate_daily_meals(bootstrap_sample_dict):
    """
    Aggregate meals for each day across all meal types, considering the union of all unique columns.
    
    :param bootstrap_sample_dict: Dictionary with meal types as keys and DataFrames as values.
    :return: DataFrame of aggregated meals for each day.
    """
    # Initialize a DataFrame to hold the aggregated meals
    aggregated_meals_df = pd.DataFrame()

    # Determine the number of days to aggregate over by finding the meal type with the maximum length
    num_days = max(len(df) for df in bootstrap_sample_dict.values())
    
    # Gather all unique columns from every meal type and initialize aggregated_meals_df with these columns
    all_columns = []
    for meals_df in bootstrap_sample_dict.values():
        all_columns.extend(meals_df.columns)
    all_columns = list(set(all_columns))
    aggregated_meals_df = pd.DataFrame(columns=all_columns)
    
    # Loop through each day and aggregate meals across all meal types for that day
    for day in range(1, num_days + 1):
        daily_aggregate = pd.Series(0, index=all_columns)
        for meal_type, meals_df in bootstrap_sample_dict.items():
            try:
                daily_meal = meals_df.loc[f'{meal_type}_{day}']
            except KeyError:
                # If a specific meal for a day is missing, skip to the next meal type
                print(f"Meal {meal_type}_{day} not found, skipping.")
                continue
            
            # Ensure daily_meal is a Series for consistent operation
            if isinstance(daily_meal, pd.DataFrame):
                daily_meal = daily_meal.iloc[0]
            
            # Aggregate the nutrient values from this meal into the daily aggregate
            for col in daily_meal.index:
                daily_aggregate[col] += daily_meal[col]
        
        # Append the daily aggregate to the aggregated meals DataFrame
        aggregated_meals_df = aggregated_meals_df.append(daily_aggregate, ignore_index=True)

    # Replace initial zeros in the aggregated meals DataFrame with NaN where no addition was performed
    #aggregated_meals_df.replace(0, pd.NA, inplace=True)
    
    # Filter out columns that remained zero after aggregation
    nonzero_columns = aggregated_meals_df.loc[:, (aggregated_meals_df != 0).any(axis=0)].columns
    nonzero_aggregated_meals = aggregated_meals_df[nonzero_columns]
    #print("Nonzero aggregated elements for the day:\n", nonzero_aggregated_meals)

    return aggregated_meals_df


def process_daily_meals(daily_meals, nutrient_targets, nutrient_fields, nutrient_data):
    """
    Process and aggregate daily meals, calculating nutrient coverage for each day, 
    and organize the results such that each nutrient's coverage across days is stored in a list.

    :param daily_meals: DataFrame containing meals, assumed to be ordered by day.
    :param nutrient_targets: Dictionary of nutrient targets.
    :param nutrient_fields: List of nutrient fields to consider.
    :param nutrient_data: DataFrame with nutrient data.
    :return: Dictionary, with each key being a nutrient and its value a list of daily coverage values.
    """
    # Initialize a dictionary with each nutrient field pointing to an empty list
    nutrient_coverage_lists = {nutrient: [] for nutrient in nutrient_fields}
    # Assuming daily_meals is structured to iterate over each "day" sequentially
    for _, meal in daily_meals.iterrows():
        # Aggregate nutrients for the meal
        meal_nutrition = calculate_meal_nutrition_from_df(meal, nutrient_data)
        # Replace newlines in the indices with a space and rename based on mapping
        # Temporarily store nutrient coverage for the current meal/day
        temp_nutrient_coverage = {}
        for nutrient in nutrient_fields:
            rdi = nutrient_targets.get(nutrient, 0)
            nutrient_amount = meal_nutrition.get(nutrient, 0)
            coverage_ratio = (nutrient_amount - rdi) / rdi if rdi > 0 else 0
            temp_nutrient_coverage[nutrient] = coverage_ratio

        # Append each nutrient's coverage for the current day to its respective list in nutrient_coverage_lists
        for nutrient, coverage in temp_nutrient_coverage.items():
            if nutrient in nutrient_coverage_lists:
                nutrient_coverage_lists[nutrient].append(coverage)
    # After processing all days/meals, return the dictionary of lists
    return nutrient_coverage_lists

def calculate_nutrient_coverage_parallel(bootstrap_sample_dict, nutrient_targets, nutrient_fields, nutrient_data):
    daily_meals = aggregate_daily_meals(bootstrap_sample_dict)
    nutrient_coverage = process_daily_meals(daily_meals, nutrient_targets, nutrient_fields, nutrient_data)
    avg_coverage_scores = {}
    for nutrient, daily_coverages in nutrient_coverage.items():
        if daily_coverages:  # Ensure the list is not empty
            avg_coverage_scores[nutrient] = sum(daily_coverages) / len(daily_coverages)
        else:
            avg_coverage_scores[nutrient] = None  # or 0, or another placeholder for nutrients without data
    return avg_coverage_scores

def calculate_overall_nutrient_coverage(bootstrap_sample_dict, nutrient_fields, nutrient_data, calorie_level='2000 cal'):    
    # Load nutrient targets for the specified calorie level
    nutrient_targets = load_nutrient_targets(calorie_level)
    # Convert nutrient targets to dictionary for easier access
    nutrient_targets_dict = nutrient_targets.to_dict()
    # Calculate average coverage scores
    avg_coverage_scores = calculate_nutrient_coverage_parallel(bootstrap_sample_dict, nutrient_targets_dict, nutrient_fields, nutrient_data)
    
    # Calculate the total score by summing all average coverage values
    total_score = sum(avg_coverage_scores.values())
    
    # Count the number of nutrients to average over, ignoring None values
    num_nutrients = sum(1 for value in avg_coverage_scores.values() if value is not None)

    # Calculate the overall average coverage score
    overall_average_coverage = total_score / num_nutrients if num_nutrients > 0 else 0

    return overall_average_coverage, avg_coverage_scores

def calculate_nutrient_density_from_meals(meal_df, nutrient_data):
    # Initialize a list to hold nutrient densities for each meal
    meal_nutrient_densities = []

    # Iterate over each meal in the DataFrame
    for _, meal_row in meal_df.iterrows():
        # Calculate the nutrition for this single meal
        meal_nutrition = calculate_meal_nutrition_from_df(meal_row, nutrient_data)
        # Calculate total calories for this meal
        total_calories = meal_nutrition['Energy (kcal)']
        
        # Calculate nutrient densities for this meal, excluding 'Energy (kcal)'
        nutrient_densities = {nutrient: meal_nutrition[nutrient] / total_calories if total_calories > 0 else 0
                              for nutrient in meal_nutrition.keys() if nutrient != 'Energy (kcal)'}
        
        meal_nutrient_densities.append(nutrient_densities)

    # Convert list of dictionaries to a DataFrame for easier calculation of average densities
    meal_nutrient_densities_df = pd.DataFrame(meal_nutrient_densities)
    average_nutrient_densities = meal_nutrient_densities_df.mean().to_dict()

    return average_nutrient_densities
    
def calculate_average_nutrient_density(meal_types, nutrient_data):
    nutrient_density_dict = {}
    real_meal_average_nutrient_densities_file = f'data/processed/real_meals/dataframes/aggregate_nutrient_densities.pkl'
    if not os.path.exists(real_meal_average_nutrient_densities_file):
        for meal_type in meal_types:
            meal_df = pd.read_pickle(f'data/processed/real_meals/dataframes/{meal_type}_sparsity_filtered.pkl')
            nutrient_density_dict[meal_type] = calculate_nutrient_density_from_meals(meal_df, nutrient_data)
        to_pickle(nutrient_density_dict, real_meal_average_nutrient_densities_file)
    else:
        nutrient_density_dict = pd.read_pickle(real_meal_average_nutrient_densities_file)
    return nutrient_density_dict

def filter_corresponding_nutrient_fields(meal_densities, nutrient_fields):
    meal_densities_filtered = {}
    for meal_type in meal_densities.keys():
        meal_densities_filtered[meal_type] = {nutrient: meal_densities[meal_type][nutrient] for nutrient in nutrient_fields if nutrient != 'Energy (kcal)'}
    return meal_densities_filtered

def compare_bootstrap_with_real_meals(bootstrap_sample_dict, nutrient_fields, nutrient_data):
    average_real_meal_densities = calculate_average_nutrient_density(bootstrap_sample_dict.keys(), nutrient_data)
    average_bootstrap_densities = {meal_type: calculate_nutrient_density_from_meals(bootstrap_sample_dict[meal_type], nutrient_data) 
                                   for meal_type in bootstrap_sample_dict}
    average_real_meal_densities = filter_corresponding_nutrient_fields(average_real_meal_densities, nutrient_fields)
    average_bootstrap_densities = filter_corresponding_nutrient_fields(average_bootstrap_densities, nutrient_fields)
    cosine_similarities = get_cos_similarity(average_bootstrap_densities, average_real_meal_densities, nutrient_fields)

    return cosine_similarities

def get_cos_similarity(dict1, dict2, keys_list):
    cosine_similarities = {}
    for primary_key in dict1.keys():
        vector1 = [dict1[primary_key].get(key, 0) for key in keys_list]
        vector2 = [dict2[primary_key].get(key, 0) for key in keys_list]
        similarity = 1 - cosine(vector1, vector2)
        cosine_similarities[primary_key] = similarity

    return cosine_similarities

def calculate_average_portions_from_df(meals_df):
    portion_totals = {}
    portion_counts = {}
    for _, meal in meals_df.iterrows():
        for food_item, portion in meal.items():
            if portion > 0:  # Consider only non-zero portions
                portion_totals[food_item] = portion_totals.get(food_item, 0) + portion
                portion_counts[food_item] = portion_counts.get(food_item, 0) + 1
                
    average_portions = {food: (portion_totals[food] / portion_counts[food]) for food in portion_totals}
    return average_portions

def min_max_scale_values(values_dict):
    all_values = [value for subtype_values in values_dict.values() for value in subtype_values.values()]
    min_value, max_value = min(all_values), max(all_values)

    scaled_values = {}
    for subtype, subtype_values in values_dict.items():
        scaled_values[subtype] = {}
        for key, value in subtype_values.items():
            scaled_value = ((value - min_value) / (max_value - min_value)) if max_value != min_value else 0
            scaled_values[subtype][key] = scaled_value

    return scaled_values

def average_deviations_by_category(scaled_deviations, food_category_mapping):
    category_deviations = {meal_type: {} for meal_type in scaled_deviations}
    for meal_type, foods in scaled_deviations.items():
        for food, deviation in foods.items():
            category = food_category_mapping.get(food)
            if category:
                if category not in category_deviations[meal_type]:
                    category_deviations[meal_type][category] = []
                category_deviations[meal_type][category].append(deviation)

    average_category_deviations = {
        meal_type: {cat: sum(devs) / len(devs) for cat, devs in categories.items()}
        for meal_type, categories in category_deviations.items()
    }
    return average_category_deviations

def group_deviations_by_category(scaled_deviations, food_category_mapping):
    category_deviations = {meal_type: {} for meal_type in scaled_deviations}

    for meal_type, foods in scaled_deviations.items():
        for food, deviation in foods.items():
            category = food_category_mapping.get(food)
            if category:
                if category not in category_deviations[meal_type]:
                    category_deviations[meal_type][category] = []
                category_deviations[meal_type][category].append(deviation)

    return category_deviations

def calculate_portion_size_analysis(bootstrap_sample, real_meals_dict):
    meal_type_deviations = {meal_type: {} for meal_type in bootstrap_sample.keys()}
    
    for meal_type, generated_meals_df in bootstrap_sample.items():
        average_portions_real = calculate_average_portions_from_df(real_meals_dict[meal_type])
        average_portions_generated = calculate_average_portions_from_df(generated_meals_df)
        
        for food, avg_portion_real in average_portions_real.items():
            if food in average_portions_generated:
                avg_portion_generated = average_portions_generated[food]
                deviation = (avg_portion_generated - avg_portion_real) / avg_portion_real if avg_portion_real > 0 else 0
                meal_type_deviations[meal_type][food] = deviation

    return meal_type_deviations

def calculate_average_category_weights(meal_type, food_category_mapping):
    real_meals = pd.read_pickle(f'data/processed/real_meals/dataframes/{meal_type}_sparsity_filtered.pkl')
    category_weights = {category: 0 for category in food_category_mapping.values()}
    total_weight = sum(real_meals.sum())

    for food in real_meals.columns:
        category = food_category_mapping.get(food)
        if category:
            category_weights[category] += real_meals[food].sum()

    average_category_weights = {category: weight / total_weight for category, weight in category_weights.items()}
    return average_category_weights

def calculate_bootstrap_category_weights(bootstrap_sample, food_category_mapping):
    bootstrap_category_weights = {meal_type: {} for meal_type in bootstrap_sample}
    
    for meal_type, df in bootstrap_sample.items():
        total_weight = df.sum().sum()
        for food in df.columns:
            category = food_category_mapping.get(food)
            if category:
                if category not in bootstrap_category_weights[meal_type]:
                    bootstrap_category_weights[meal_type][category] = 0
                bootstrap_category_weights[meal_type][category] += df[food].sum()
        bootstrap_category_weights[meal_type] = {category: weight / total_weight for category, weight in bootstrap_category_weights[meal_type].items()}
        
    return bootstrap_category_weights

def calculate_category_balance_similarity_scores(bootstrap_sample):
    mapping_df = pd.read_pickle('data/mappings/food_ingredient_map.pkl')
    food_category_mapping = dict(zip(mapping_df['Food code'], mapping_df['WWEIA Main Category description']))
    bootstrap_category_weights = calculate_bootstrap_category_weights(bootstrap_sample, food_category_mapping)
    similarity_scores = {meal_type: {} for meal_type in bootstrap_category_weights}
    
    for meal_type, categories in bootstrap_category_weights.items():
        average_category_weights = calculate_average_category_weights(meal_type, food_category_mapping)
        for category, bootstrap_proportion in categories.items():
            if bootstrap_proportion > 0:
                real_meal_proportion = average_category_weights.get(category, 0)
                if real_meal_proportion != 0:
                    score = abs(bootstrap_proportion / real_meal_proportion)
                else:
                    score = 0  # Category is present in bootstrap but not in real meals
                similarity_scores[meal_type][category] = score

    return similarity_scores

def calculate_food_coverage(bootstrap_sample_dict, real_meals_dict):
    food_coverage_dict = {meal_type: {} for meal_type in bootstrap_sample_dict.keys()}
    for meal_type, bootstrap_df in bootstrap_sample_dict.items():
        unique_foods_generated = set(bootstrap_df.columns[bootstrap_df.sum(axis=0) > 0])
        unique_foods_possible = set(real_meals_dict[meal_type].columns)
        food_coverage_dict[meal_type] = len(unique_foods_generated & unique_foods_possible) / len(unique_foods_possible)
    return food_coverage_dict

def calculate_relative_meal_coverage(generated_meals, real_meals):
    unique_generated_meals = len(set(map(tuple, generated_meals.values.tolist())))
    unique_real_meals = len(set(map(tuple, real_meals.values.tolist())))
    total_possible_meals = 2 ** len(real_meals.columns) - 1

    coverage_ratio_generated = unique_generated_meals / total_possible_meals
    coverage_ratio_real = unique_real_meals / total_possible_meals
    relative_coverage = coverage_ratio_generated / coverage_ratio_real if coverage_ratio_real > 0 else 0

    return relative_coverage

# Example of adjusting the calculate_meal_coverage function within the existing code structure
def calculate_meal_coverage_adjusted(bootstrap_sample_dict, real_meals_dict):
    relative_coverage_dict = {}
    for meal_type in bootstrap_sample_dict.keys():
        generated_meals = bootstrap_sample_dict[meal_type]
        real_meals = real_meals_dict[meal_type]
        relative_coverage = calculate_relative_meal_coverage(generated_meals, real_meals)
        relative_coverage_dict[meal_type] = relative_coverage
    return relative_coverage_dict

def calculate_meal_coverage(bootstrap_sample_dict, real_meals_dict):
    meal_coverage_dict = {meal_type: {} for meal_type in bootstrap_sample_dict.keys()}
    for meal_type, bootstrap_df in bootstrap_sample_dict.items():
        unique_meals = len(set(map(str, (bootstrap_df > 0).values.tolist())))
        all_possible_meals = 2 ** len(set(real_meals_dict[meal_type].columns)) - 1
        meal_coverage_dict[meal_type] = unique_meals / all_possible_meals
    return meal_coverage_dict

def calculate_meal_diversity_pielous_evenness(bootstrap_sample_dict):
    meal_diversity_dict = {meal_type: {} for meal_type in bootstrap_sample_dict.keys()}
    for meal_type, bootstrap_df in bootstrap_sample_dict.items():
        binary_matrix = (bootstrap_df > 0).astype(int)
        
        component_presence_freqs = binary_matrix.mean(axis=0)
        total_freq = component_presence_freqs.sum()
        component_probs = component_presence_freqs / total_freq if total_freq > 0 else component_presence_freqs
        
        # Calculate Shannon entropy (H') from these probabilities
        H_prime = entropy(component_probs, base=2)
        S = len(component_probs)
        H_max = np.log2(S) if S > 0 else 0
        
        # Calculate Pielou's Evenness (J')
        J_prime = H_prime / H_max if H_max > 0 else np.nan  # Use NaN if H_max is 0
        
        meal_diversity_dict[meal_type] = J_prime

    return meal_diversity_dict

def calculate_meal_diversity(bootstrap_sample_dict):
    meal_diversity_dict = {meal_type: {} for meal_type in bootstrap_sample_dict.keys()}
    for meal_type, bootstrap_df in bootstrap_sample_dict.items():
        meal_freqs = bootstrap_df.mean(axis=0) / bootstrap_df.mean(axis=0).sum()
        meal_diversity_dict[meal_type] = entropy(meal_freqs, base=2)
    return meal_diversity_dict

def calculate_food_overlap_similarity_jaccard(bootstrap_sample_dict, real_meals_dict):
    meal_realism_dict = {meal_type: {} for meal_type in bootstrap_sample_dict.keys()}
    for meal_type, bootstrap_df in bootstrap_sample_dict.items():
        active_generated_ingredients = bootstrap_df.columns[bootstrap_df.max() > 0]
        generated_ingredients_set = set(active_generated_ingredients)
        real_ingredients_set = set(real_meals_dict[meal_type].columns)

        intersection = generated_ingredients_set.intersection(real_ingredients_set)
        union = generated_ingredients_set.union(real_ingredients_set)
        jaccard_index = len(intersection) / len(union) if len(union) > 0 else 0

        meal_realism_dict[meal_type] = jaccard_index
    return meal_realism_dict

def calculate_meal_realism(bootstrap_sample_dict, real_meals_dict):
    meal_realism_dict = {meal_type: {} for meal_type in bootstrap_sample_dict.keys()}
    for meal_type, bootstrap_df in bootstrap_sample_dict.items():
        all_foods = set(real_meals_dict[meal_type].columns) | set(bootstrap_df.columns)
        expanded_real_freqs = [real_meals_dict[meal_type][food].mean() if food in real_meals_dict[meal_type].columns else 1e-10 for food in all_foods]
        expanded_generated_freqs = [bootstrap_df[food].mean() if food in bootstrap_df.columns else 1e-10 for food in all_foods]

        total_real = sum(expanded_real_freqs)
        total_generated = sum(expanded_generated_freqs)
        expanded_real_freqs = [freq / total_real for freq in expanded_real_freqs]
        expanded_generated_freqs = [freq / total_generated for freq in expanded_generated_freqs]

        epsilon = 1e-10
        expanded_real_freqs = [freq + epsilon for freq in expanded_real_freqs]
        expanded_generated_freqs = [freq + epsilon for freq in expanded_generated_freqs]

        meal_realism_dict[meal_type] = entropy(expanded_real_freqs, expanded_generated_freqs, base=2)
    return meal_realism_dict

def average_nested_dict_values(data):
    averages = {}
    for key, nested_dict in data.items():
        average = sum(nested_dict.values()) / len(nested_dict) if nested_dict else 0
        averages[key] = average
    return averages

def calculate_meal_metrics(bootstrap_sample_dict, real_meals_dict, food_category_mapping, num_days):    
    """
    # Meal metrics
    aggregated_nutrition = aggregate_nutrition_per_day(bootstrap_sample_dict, num_days)
    macronutrient_coverage = calculate_nutrient_coverage(aggregated_nutrition, MACRONUTRIENTS)
    micronutrient_coverage = calculate_nutrient_coverage(aggregated_nutrition, MICRONUTRIENTS)
    macronutrient_density = calculate_nutrient_density(aggregated_nutrition, MACRONUTRIENTS)
    micronutrient_density = calculate_nutrient_density(aggregated_nutrition, MICRONUTRIENTS)
    """
    
    nutrient_data = pd.read_pickle('data/mappings/food_to_nutrient.pkl')
    nutrient_data.set_index('Food code', inplace=True)
    nutrient_data = nutrient_data.apply(pd.to_numeric, errors='coerce').fillna(0)

    macronutrient_coverage, macronutrient_coverage_by_nutrient = calculate_overall_nutrient_coverage(bootstrap_sample_dict, MACRONUTRIENTS, nutrient_data)
    micronutrient_coverage, micronutrient_coverage_by_nutrient = calculate_overall_nutrient_coverage(bootstrap_sample_dict, MICRONUTRIENTS, nutrient_data)
    macronutrient_density = compare_bootstrap_with_real_meals(bootstrap_sample_dict, MACRONUTRIENTS, nutrient_data)
    micronutrient_density = compare_bootstrap_with_real_meals(bootstrap_sample_dict, MICRONUTRIENTS, nutrient_data)
    
    portion_size_deviation = calculate_portion_size_analysis(bootstrap_sample_dict, real_meals_dict)
    average_category_deviations = average_deviations_by_category(portion_size_deviation, food_category_mapping)
    food_category_balance = calculate_category_balance_similarity_scores(bootstrap_sample_dict)
    
    portion_size_deviation_bootstrap_average = average_nested_dict_values(portion_size_deviation)
    food_category_balance_bootstrap_average = average_nested_dict_values(food_category_balance)
    
    #meal_coverage = calculate_meal_coverage_adjusted(bootstrap_sample_dict, real_meals_dict)
    food_coverage = calculate_food_coverage(bootstrap_sample_dict, real_meals_dict)
    meal_diversity = calculate_meal_diversity_pielous_evenness(bootstrap_sample_dict)
    meal_realism = calculate_food_overlap_similarity_jaccard(bootstrap_sample_dict, real_meals_dict)
    
    # Aggregate all metrics into a structured dictionary
    metrics = {
        'Food Coverage': food_coverage,
        #'Meal Coverage': meal_coverage,
        'Meal Diversity': meal_diversity,
        #'Food Overlap Similarity': meal_realism,
        'Meal Realism': meal_realism,
        'Macronutrient Density': macronutrient_density,
        'Micronutrient Density': micronutrient_density,
        'Macronutrient Coverage': macronutrient_coverage,
        'Micronutrient Coverage': micronutrient_coverage,
        'Portion Size Deviation': portion_size_deviation_bootstrap_average,
        'Food Category Balance': food_category_balance_bootstrap_average,
    }
    plot_metrics = {
        "Average Category Deviations": average_category_deviations,
        "Average Category Balance": food_category_balance,
        "Average Macronutrient Coverage": macronutrient_coverage_by_nutrient,
        "Average Micronutrient Coverage": micronutrient_coverage_by_nutrient
    }
    
    return metrics, plot_metrics
    