import os
import re
import sys
import pandas as pd
import numpy as np
from sklearn.neighbors import LocalOutlierFactor
from prettytable import PrettyTable 

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.helpers import *
from training.train_portion_model import *
from training.train_presence_model import *
        
def create_meals(data_subset, day_prefix):
    grouped = data_subset.groupby('SEQN')
    meals = []
    for _, group in grouped:
        id = group[f'SEQN'].iloc[0]
        food_codes = group[f'{day_prefix}IFDCD'].tolist()
        gram_portions = group[f'{day_prefix}IGRMS'].tolist()
        day_eaten = group[f'{day_prefix}DAY'].iloc[0]  # Assuming the same for all rows in the group.
        food_source = group[f'{day_prefix}FS'].tolist()
        # TODO: ADD DEMOGRAPHIC DATA
        meals.append(Meal(id, food_codes, gram_portions, day_eaten, food_source))
    return meals

def standardize_column_names(df, day_prefix):
    standardized_columns = {col: col.replace(day_prefix, '') for col in df.columns if col.startswith(day_prefix)}
    df.rename(columns=standardized_columns, inplace=True)
    return df

def get_lowest_food_codes(discontinued_id_map, code):
    code_info = discontinued_id_map.get(code)
    # if no entry for this code, it's a lowest code
    if code_info is None:
        return [{'code': code, 'revision': 0}]
    lowest_codes = []
    for new_code in code_info['code']: 
        lower_codes = get_lowest_food_codes(discontinued_id_map, new_code)
        if not lower_codes:  # if no lower codes, new code is lowest
            lowest_codes.append({'code': new_code, 'revision': code_info['revision']})
        else:
            lowest_codes.extend(lower_codes)  # otherwise add lower codes found

    return lowest_codes

def process_diet_data(data, day_prefix):
    breakfast_values = [1, 10]
    lunch_values = [2, 11]
    dinner_values = [3, 4, 14]

    breakfast_data = data[data[f'{day_prefix}_030Z'].isin(breakfast_values)]
    lunch_data = data[data[f'{day_prefix}_030Z'].isin(lunch_values)]
    dinner_data = data[data[f'{day_prefix}_030Z'].isin(dinner_values)]

    breakfast_meals = create_meals(breakfast_data, day_prefix)
    lunch_meals = create_meals(lunch_data, day_prefix)
    dinner_meals = create_meals(dinner_data, day_prefix)

    return breakfast_meals, lunch_meals, dinner_meals

def generate_discontinued_id_map(discontinued_urls, cache_dir, preprocess_log):
    discontinued_id_map = {}
    for file_url in discontinued_urls:
        try:
            filename = os.path.basename(file_url)
            local_filepath = os.path.join(cache_dir, filename)
            if not os.path.exists(local_filepath):
                preprocess_log.info(f"Downloading {filename} from {file_url}...")
                download_file(file_url, local_filepath, preprocess_log)
            else:
                preprocess_log.info(f"Using cached file: {filename}")

            if local_filepath.endswith(".xlsx"):
                discontinued_df = pd.read_excel(local_filepath, engine='openpyxl', skiprows=1)
            else:
                raise ValueError(f"Unsupported file format for {local_filepath}")

            # Get the first and fourth columns as the old and new food codes
            old_code_column = discontinued_df.columns[0]
            new_code_column = discontinued_df.columns[3]
            discontinued_id_column = discontinued_df.columns[2] 

            for _, row in discontinued_df.iterrows():
                discontinued_id = row[discontinued_id_column]
                old_code = row[old_code_column]
                new_codes = row[new_code_column]
                # Handle the case where there are multiple new codes for a single old code.
                if discontinued_id == 2 and isinstance(new_codes, str):
                    new_codes = re.split(r'[;\n]', new_codes)
                    new_codes = [int(code.strip()) for code in new_codes if code.strip()]
                elif discontinued_id == 5 or discontinued_id == 1:
                    continue
                else: # discontinued codes 3 or 4
                    new_codes = [int(new_codes)]

                discontinued_id_map[old_code] = {
                    'code': new_codes,  
                    'revision': discontinued_id, 
                }
        except Exception as e:
            preprocess_log.error(f"Error processing file {filename}: {e}")
            continue
        
    to_pickle(discontinued_id_map, 'data/mappings/discontinued_food_codes.pkl')
    return discontinued_id_map

def process_row(row, day_prefix, discontinued_id_map):
    food_code = row[f'{day_prefix}IFDCD']
    lowest_codes = get_lowest_food_codes(discontinued_id_map, food_code)
    
    if not lowest_codes:
        return [row]
    
    updated_rows = []
    for code_info in lowest_codes:
        new_code = code_info['code']
        discontinued_id = code_info['revision']

        new_row = row.copy()
        if discontinued_id == 1 or discontinued_id == 5:
            # for dropped (1) or revised (5) items, we keep the original code.
            updated_rows.append(row)
        elif discontinued_id == 2:
            # for expanded (2) items, we create new rows with distributed gram weights
            portion = row[f'{day_prefix}IGRMS'] / len(lowest_codes)  # Divide equally among new codes
            new_row[f'{day_prefix}IFDCD'] = new_code
            new_row[f'{day_prefix}IGRMS'] = portion
            updated_rows.append(new_row)
        else:
            # for consolidated (3) or renumbered (4) items, we replace with the new code
            new_row[f'{day_prefix}IFDCD'] = new_code
            updated_rows.append(new_row)
    return updated_rows

def update_food_codes(day_prefix, data, discontinued_id_map, preprocess_log):
    original_unique_food_codes = data[f'{day_prefix}IFDCD'].unique()
    original_food_code_count = data[f'{day_prefix}IFDCD'].nunique()
    
    updated_data_list = data.apply(lambda row: process_row(row, day_prefix, discontinued_id_map), axis=1)
    flat_list = [item for sublist in updated_data_list for item in sublist]
    updated_data = pd.DataFrame(flat_list)

    updated_food_code_count = updated_data[f'{day_prefix}IFDCD'].nunique()
    overall_difference = updated_food_code_count - original_food_code_count
    preprocess_log.info(f"Original unique food codes:       {original_food_code_count}")
    preprocess_log.info(f"Updated unique food codes:        {updated_food_code_count}")
    preprocess_log.info(f"Overall difference in food codes: {overall_difference}")
    
    return updated_data, original_unique_food_codes

def merge_weight_columns(df):
    if 'WTDRD1' in df.columns:
        df['WTD1'] = df['WTDRD1']
        df.drop(['WTDRD1'], axis=1, inplace=True)
    if 'WTDRD1PP' in df.columns:
        df['WTD1'] = df['WTDRD1PP']
        df.drop(['WTDRD1PP'], axis=1, inplace=True)
    if 'WTDR2D' in df.columns:
        df['WTD2'] = df['WTDR2D']
        df.drop(['WTDR2D'], axis=1, inplace=True)
    if 'WTDR2DPP' in df.columns:
        df['WTD2'] = df['WTDR2DPP']
        df.drop(['WTDR2DPP'], axis=1, inplace=True)
    return df

def load_and_process_meal_data(meal_survey_urls, discontinued_id_map, cache_dir, preprocess_log):
    all_updated_data, original_unique_food_codes = [], []
    breakfast_meals, lunch_meals, dinner_meals = [], [], []
    breakfast_raw_meals, lunch_raw_meals, dinner_raw_meals = [], [], []
    for url in meal_survey_urls:
        
        try:
            preprocess_log.info(f"Processing data from URL: {url}")
            meal_data = fetch_data(url, cache_dir, preprocess_log)
            preprocess_log.info(f"Unique Respondents: {meal_data['SEQN'].nunique()}")
            preprocess_log.info(f"Number of Records: {len(meal_data)}")
            day_prefix = 'DR1' if 'DR1' in url else 'DR2'

            breakfast_raw, lunch_raw, dinner_raw = process_diet_data(meal_data, day_prefix)
            breakfast_raw_meals.extend(breakfast_raw)
            lunch_raw_meals.extend(lunch_raw)
            dinner_raw_meals.extend(dinner_raw)
            
            preprocess_log.info(f"Raw meals: {len(breakfast_raw) + len(lunch_raw) + len(dinner_raw)} - {len(breakfast_raw)} breakfast, {len(lunch_raw)} lunch, and {len(dinner_raw)} dinner from {url}")
            updated_meal_data, subset_unique_food_codes = update_food_codes(day_prefix, meal_data, discontinued_id_map, preprocess_log)
            original_unique_food_codes.extend(subset_unique_food_codes)
            updated_meal_data = updated_meal_data[updated_meal_data[f'{day_prefix}DRSTZ'] == 1]  # diet recall status code is 1
            updated_meal_data = updated_meal_data[updated_meal_data['DRABF'] == 2]  # breast-fed infant is 'No'
            
            breakfast_temp, lunch_temp, dinner_temp = process_diet_data(updated_meal_data, day_prefix)
            preprocess_log.info(f"Processed meals: {len(breakfast_temp) + len(lunch_temp) + len(dinner_temp)} - {len(breakfast_temp)} breakfast, {len(lunch_temp)} lunch, and {len(dinner_temp)} dinner from {url}")
            # Merge the weight columns before standardizing column names
            updated_meal_data = merge_weight_columns(updated_meal_data)
            updated_meal_data = standardize_column_names(updated_meal_data, day_prefix)
            #updated_meal_data.rename(columns=standardized_columns, inplace=True)
            all_updated_data.append(updated_meal_data)
            breakfast_meals.extend(breakfast_temp)
            lunch_meals.extend(lunch_temp)        
            dinner_meals.extend(dinner_temp)
            
        except Exception as e:
            preprocess_log.error(f"Error processing meal data from {url}: {e}")
            continue
        
    raw_df_dir = 'data/raw/meals/dataframes/'
    os.makedirs(raw_df_dir, exist_ok=True)
    to_pickle(breakfast_raw_meals, raw_df_dir+'breakfast_raw.pkl')
    to_pickle(lunch_raw_meals, raw_df_dir+'lunch_raw.pkl')
    to_pickle(dinner_raw_meals, raw_df_dir+'dinner_raw.pkl')
    
    preprocess_log.info(f"Raw unique food codes: {len(set(original_unique_food_codes))}")
    combined_meal_data = pd.concat(all_updated_data, ignore_index=True)
    to_pickle(combined_meal_data, 'data/processed/real_meals/dataframes/combined_food_codes_standardized.pkl')
    to_pickle(breakfast_meals, 'data/processed/real_meals/breakfast_meals.pkl')
    to_pickle(lunch_meals, 'data/processed/real_meals/lunch_meals.pkl')
    to_pickle(dinner_meals, 'data/processed/real_meals/dinner_meals.pkl')
    
    return combined_meal_data, breakfast_meals, lunch_meals, dinner_meals

def remove_outliers(df, meal_type, preprocess_log, contamination=0.003):
    try:
        preprocess_log.info("Removing outliers...")
        X = df.values
        lof = LocalOutlierFactor(n_neighbors=20, contamination=contamination, novelty=False)
        # Where -1 indicates outliers and 1 indicates inliers
        lof.fit_predict(X)
        # Calculate LOF scores (negative_outlier_factor_ is opposite, more negative means more outlier)
        lof_scores = -lof.negative_outlier_factor_  # We negate to make positive, higher means more outlier
        cutoff_score = np.percentile(lof_scores, 99.7)
        # Identifying indexes to keep (inliers and low-scoring outliers)
        inlier_indexes = (lof_scores <= cutoff_score)
        num_outliers = np.sum(~inlier_indexes)
        preprocess_log.info(f"Number of outliers removed: {num_outliers}")
        inlier_indexes = (lof_scores <= cutoff_score)
        filtered_df = df[inlier_indexes]
        to_pickle(filtered_df, f'data/processed/real_meals/dataframes/{meal_type}_outliers_removed.pkl')
    except Exception as e:
        preprocess_log.error("Error in removing outliers: " + str(e))
    return filtered_df

def bootstrap_confidence_interval_filter(df, meal_type, preprocess_log, n_iterations=1000, ci_threshold=0.95):
    try:
        preprocess_log.info("Applying bootstrap confidence interval filter...")
        binary_df = df.gt(0).astype(int)
        bootstrapped_means = np.zeros((n_iterations, binary_df.shape[1]))
        
        for i in range(n_iterations):
            sample = binary_df.sample(frac=1.0, replace=True)
            bootstrapped_means[i, :] = sample.mean()
        lower_bounds = np.percentile(bootstrapped_means, (1 - ci_threshold) / 2 * 100, axis=0)

        # Determine which food codes are above the lower bound of the CI
        mask = np.mean(binary_df, axis=0) > lower_bounds
        filtered_df = df.loc[:, mask]
        
        foods_retained = np.sum(mask)
        foods_removed = df.shape[1] - foods_retained
        preprocess_log.info(f"Retained {foods_retained} foods, removed {foods_removed} foods")
        
        num_rows_before = filtered_df.shape[0]
        filtered_df = filtered_df[filtered_df.any(axis=1)]
        num_rows_after = filtered_df.shape[0]
        num_rows_removed = num_rows_before - num_rows_after
        preprocess_log.info(f"Removed {num_rows_removed} meals with all zero values post-filtration")

        to_pickle(filtered_df, f'data/processed/real_meals/dataframes/{meal_type}_bci_filtered.pkl')
    except Exception as e:
        preprocess_log.error("Error in bootstrap confidence interval filtering: " + str(e))

def train_models_by_sparsity(data, meal_type, preprocess_log):
    outdir = f'data/processed/real_meals/dataframes/{meal_type}'

    if not os.path.exists(f'{outdir}_sparsity_filtered.pkl'):
        foods_per_sparse, meals_per_sparse = [], []
        fold_macro_results, fold_micro_results = [], []
        test_macro_results, test_micro_results = [], []
        
        config = load_config('config.json')
        for percentage in [i  for i in range(10, 100, 5)]:
            wb_group_name = f"sparsity_{percentage}"

            respath = construct_save_path(meal_type, 'presence_model', 'results', config)
            respath += f'sparsity_{percentage}/'
            os.makedirs(respath, exist_ok=True)
            logpath = construct_save_path(meal_type, 'presence_model', 'logging', config)
            logpath += f'sparsity_{percentage}'
            sparsity_logfile = f'{logpath}_training.log'
            sparsity_model_train_log = setup_logger(f'sparsity_{percentage}', sparsity_logfile)
            reduced_data = remove_columns_and_rows(data, percentage)
            meals_per_sparse.append(reduced_data.shape[0])
            foods_per_sparse.append(reduced_data.shape[1])
            sparsity_model_train_log.info(f'Dataset reduced to {reduced_data.shape[0]} meals and {reduced_data.shape[1]} foods')

            log_model_params(config, 'presence_model', sparsity_model_train_log)
            sparsity_filtration_path = f'data/processed/real_meals/dataframes/sparsity_filtration/{meal_type}/'
            os.makedirs(sparsity_filtration_path, exist_ok=True)
            to_pickle(reduced_data, sparsity_filtration_path + f'{meal_type}_sparse_{int(percentage)}.pkl')
            preprocess_log.info(f"Training model at {int(percentage)}% sparse columns removed...")
            if not os.path.exists(f'{respath}result.pkl'):
                model = PresenceModel(reduced_data.shape[1], meal_type, 'config.json')
                model.results_save_path += f'sparsity_{int(percentage)}/'
                model.model_save_path += f'sparsity_{int(percentage)}/'
                fold_macro_metrics, fold_micro_metrics, test_metrics = train_presence_model(reduced_data, meal_type, wb_group_name, sparsity_model_train_log, config)
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
    else:
        preprocess_log.info(f'{outdir}_sparsity_filtered.pkl already exists, using cached data')

def preprocessing(meal_type, preprocess_log):
    try:
        preprocess_log.info(f"Preprocessing for meal type: {meal_type}")
        outlier_file = f'data/processed/real_meals/dataframes/{meal_type}_outliers_removed.pkl'
        bci_file = f'data/processed/real_meals/dataframes/{meal_type}_bci_filtered.pkl'
        sparsity_file = f'data/processed/real_meals/dataframes/{meal_type}_sparsity_filtered.pkl'

        if not os.path.exists(outlier_file):
            meal_df = pd.read_pickle(f'data/processed/real_meals/dataframes/{meal_type}_food_codes_standardized.pkl')
            outlier_df  = remove_outliers(meal_df, meal_type, preprocess_log)
        else:
            preprocess_log.info(f'{outlier_file} exists, using cached results.')
            outlier_df = pd.read_pickle(outlier_file)
        
        if not os.path.exists(bci_file):
            bootstrap_confidence_interval_filter(outlier_df, meal_type, preprocess_log)
        else:
            preprocess_log.info(f'{bci_file} exists, using cached results.')
            bci_df = pd.read_pickle(bci_file)
            
        if not os.path.exists(sparsity_file):
            train_models_by_sparsity(bci_df, meal_type, preprocess_log)
        else:
            preprocess_log.info(f'{sparsity_file} exists, using cached results.')

    except Exception as e:
        preprocess_log.error("Error in preprocessing: " + str(e))
        
def create_meal_df(meal_type, preprocess_log):
    try:
        preprocess_log.info(f"Creating meal DataFrame for: {meal_type}")
        meal_list = pd.read_pickle(f'data/processed/real_meals/{meal_type}_meals.pkl')
        meal_data_dicts = []
        for meal in meal_list:
            meal_dict = dict(zip(meal.food_codes, meal.gram_portions))
            meal_data_dicts.append(meal_dict)
        meal_df = pd.DataFrame(meal_data_dicts)
        meal_df.fillna(0, inplace=True)
        to_pickle(meal_df, f'data/processed/real_meals/dataframes/{meal_type}_food_codes_standardized.pkl')
    except Exception as e:
        preprocess_log.error("Error in creating meal DataFrame: " + str(e))

def preprocessing_summary(meal_types, processing_filetypes, preprocess_log):
    preprocess_log.info(f"="*66 + f" Preprocessing Summary " + f"="*67)
    summary_table = PrettyTable()
    summary_table.field_names = ["Processing Step"] + [f"{meal_type.capitalize()} Meals" for meal_type in meal_types] + ["Unique Foods"] + [f"{meal_type.capitalize()} Foods" for meal_type in meal_types] 
    for processing_step in processing_filetypes:
        process_meals, process_foods = [], []
        total_foods = []
        for meal_type in meal_types:
            if processing_step == 'raw':
                data = pd.read_pickle(f'data/raw/meals/dataframes/{meal_type}_raw.pkl')
                meal_type_foods = []
                for meal in data:
                    meal_type_foods.extend(meal.food_codes)
                unique_foods = np.unique(meal_type_foods).tolist()
                process_meals.append(len(data))
                process_foods.append(len(unique_foods))
                total_foods.extend(unique_foods)
            else:
                data = pd.read_pickle(f'data/processed/real_meals/dataframes/{meal_type}_{processing_step}.pkl')
                unique_foods = data.columns
                process_meals.append(data.shape[0])
                process_foods.append(data.shape[1])
                total_foods.extend(np.unique(unique_foods).tolist())
        total_foods = len(np.unique(total_foods))
        summary_table.add_row([processing_step] + process_meals + [total_foods] + process_foods) 

    table_string = summary_table.get_string()
    for line in table_string.splitlines():
        preprocess_log.info(line)

def standardize_food_codes(preprocess_log):
    meal_survey_urls = [
        "https://wwwn.cdc.gov/Nchs/Nhanes/2013-2014/DR1IFF_H.XPT", # 2013-2014 Day 1
        "https://wwwn.cdc.gov/Nchs/Nhanes/2013-2014/DR2IFF_H.XPT", # 2013-2014 Day 2
        "https://wwwn.cdc.gov/Nchs/Nhanes/2015-2016/DR1IFF_I.XPT", # 2015-2016 Day 1
        "https://wwwn.cdc.gov/Nchs/Nhanes/2015-2016/DR2IFF_I.XPT", # 2015-2016 Day 2
        "https://wwwn.cdc.gov/Nchs/Nhanes/2017-2018/P_DR1IFF.XPT", # 2017-2020 Day 1
        "https://wwwn.cdc.gov/Nchs/Nhanes/2017-2018/P_DR2IFF.XPT", # 2017-2020 Day 2
    ]
    discontinued_urls = [
        "https://www.ars.usda.gov/ARSUserFiles/80400530/apps/Discontinued_Codes_between_FNDDS_2011-2012_and_FNDDS_2013-2014.xlsx",              # 2013-2014
        "https://www.ars.usda.gov/ARSUserFiles/80400530/apps/Discontinued_Food_Codes_between_FNDDS_2013-2014_and_FNDDS_2015-2016_7_19_18.xlsx", # 2015-2016
        "https://www.ars.usda.gov/ARSUserFiles/80400530/apps/Discontinued_Food_Codes_between_FNDDS_2017-2018_and_FNDDS_2019-2020_013123.xlsx",  # 2017-2020
    ]
        
    cache_dir = "data/raw/meals/"
    os.makedirs(cache_dir, exist_ok=True)
    if not os.path.exists('data/mappings/discontinued_food_codes.pkl'):
        discontinued_id_map = generate_discontinued_id_map(discontinued_urls, cache_dir+'/discontinued', preprocess_log)
    else:
        preprocess_log.info("File exists, loading discontinued ID mapping")
        discontinued_id_map = pd.read_pickle('data/mappings/discontinued_food_codes.pkl')    

    if not os.path.exists('data/processed/real_meals/dataframes/combined_food_codes_standardized.pkl'):
        combined_meal_data, breakfast_meals, lunch_meals, dinner_meals = load_and_process_meal_data(meal_survey_urls, discontinued_id_map, cache_dir, preprocess_log)
    else:
        combined_meal_data  = pd.read_pickle("data/processed/real_meals/dataframes/combined_food_codes_standardized.pkl")
        breakfast_meals     = pd.read_pickle("data/processed/real_meals/breakfast_meals.pkl")
        lunch_meals         = pd.read_pickle("data/processed/real_meals/lunch_meals.pkl")
        dinner_meals        = pd.read_pickle("data/processed/real_meals/dinner_meals.pkl")   

    preprocess_log.info(f"Processed a total of {len(breakfast_meals)} breakfast meals, {len(lunch_meals)} lunch meals, and {len(dinner_meals)} dinner meals")
    unique_food_codes = combined_meal_data['IFDCD'].nunique()
    preprocess_log.info(f"Processed unique food codes: {unique_food_codes}")
        
def main():
    configure_gpu(-1)
    preprocess_log = setup_logger('preprocess_log', 'logs/preprocessing/preprocessing.log')
    
    try:
        preprocess_log.info("Creating dataset...")
        standardize_food_codes(preprocess_log)
        
        meal_types = ['breakfast', 'lunch', 'dinner']
        meal_types = ['lunch', 'dinner']
        processing_types = ['raw', 'food_codes_standardized', 'outliers_removed', 'bci_filtered', 'sparsity_filtered']
        for meal_type in meal_types:
            if not os.path.exists(f'data/processed/real_meals/dataframes/{meal_type}_food_codes_standardized.pkl'):
                create_meal_df(meal_type, preprocess_log)
            preprocessing(meal_type, preprocess_log)
        preprocessing_summary(meal_types, processing_types, preprocess_log)
        
    except Exception as e:
        preprocess_log.error(f"An error occurred: {e}", exc_info=True)

if __name__ == "__main__":
    main()