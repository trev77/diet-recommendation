import os
import sys
import random 
import scipy.stats as stats
from statsmodels.stats.multitest import multipletests

from sklearn.manifold import TSNE
from sklearn.cluster import DBSCAN
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import pairwise_distances

from prettytable import PrettyTable
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.helpers import *
from utils.models import *
from utils.embedding_utils import *
from training.train_embedding_model import *
from evaluation.evaluate_generation import *

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

usda_recommended_serving_size_mapping = {
    'Alcoholic Beverages': 14,
    'Baby Foods and Formulas': 100,
    'Beverages': 240,
    'Condiments and Sauces': 15,
    'Fats and Oils': 14,
    'Fruit': 150,
    'Grains': 45,
    'Milk and Dairy': 244,
    'Mixed Dishes': 200,
    'Other': 28,  # General placeholder
    'Protein Foods': 85,
    'Snacks and Sweets': 30,
    'Sugars': 4,
    'Vegetables': 85,
    'Water': 240  # Considering the liquid state
}

simplified_serving_size_mapping = {
    'Alcoholic Beverages': 30,
    'Baby Foods and Formulas': 28,
    'Beverages': 30,
    'Condiments and Sauces': 28,
    'Fats and Oils': 14,
    'Fruit': 28,
    'Grains': 28,
    'Milk and Dairy': 28,
    'Mixed Dishes': 28,
    'Other': 28,
    'Protein Foods': 28,
    'Snacks and Sweets': 28,
    'Sugars': 28,
    'Vegetables': 28,
    'Water': 30  # Considering the liquid state
}

def add_pvalue_bracket(ax, x1, x2, max_data_value, p_value):
    dy = 0.05 * max_data_value  # Dynamically adjust the height of the bracket
    y = max_data_value + dy / 2  # Position y above all boxplots with some buffer

    if p_value == 'N/A':
        p_text = f"$p = N/A$"
    elif p_value <= 0:
        p_text = "p = 0"  # Adjusted for clarity
    elif p_value < 0.001:
        exponent = np.log10(p_value)
        if not np.isfinite(exponent):
            exponent = -np.inf
        exponent = max(exponent, -300)
        p_text = f"$p < 10^{{{int(exponent)}}}$"
    else:
        p_text = f"$p = {p_value:.3f}$"

    ax.plot([x1, x1, x2, x2], [y - dy / 2, y, y, y - dy / 2], color='black', lw=1)
    ax.text((x1 + x2) / 2, y + dy / 2, p_text, ha='center', va='bottom', fontsize=12)

    
def plot_category_box_and_whisker(data, metric, corrected_pvals):
    meal_types = ['breakfast', 'lunch', 'dinner']
    top_categories = ['Beverages', 'Fats and Oils', 'Fruit', 'Grains', 'Milk and Dairy', 'Protein Foods', 'Vegetables']
    fig, axes = plt.subplots(nrows=len(meal_types), ncols=1, figsize=(15, 20), sharex=True)

    for ax, meal_type in zip(axes, meal_types):
        positions_real = np.arange(len(top_categories)) * 2 - 0.2
        positions_generated = np.arange(len(top_categories)) * 2 + 0.2

        real_data = [data['Real'][meal_type].get(category, []) for category in top_categories]
        generated_data = [data['Generated'][meal_type].get(category, []) for category in top_categories]
        bp_real = ax.boxplot(real_data, positions=positions_real, widths=0.4, patch_artist=True, boxprops=dict(facecolor="red"), showfliers=True)
        bp_generated = ax.boxplot(generated_data, positions=positions_generated, widths=0.4, patch_artist=True, boxprops=dict(facecolor="blue"), showfliers=True)

        for i, category in enumerate(top_categories):
            corrected_p = corrected_pvals.get(meal_type, {}).get(category, "N/A")
            max_data_value = max([max(data) if data else 0 for data in [real_data[i], generated_data[i]]])  # Find max value for positioning
            add_pvalue_bracket(ax, positions_real[i], positions_generated[i], max_data_value, corrected_p)

        ax.set_ylabel(metric, fontsize=14)
        ax.set_title(f'{meal_type.capitalize()} Meals', loc='left', fontsize=18, fontweight='bold')
        ax.set_xticks(np.arange(len(top_categories)) * 2)
        ax.set_xticklabels(top_categories, rotation=45, ha="right", fontsize=12)
        ax.grid(True)
        ax.set_ylim(-0.1, ax.get_ylim()[1]*1.1)  # Adjust y-axis to ensure full visibility

        # Place the legend on the bottom left of each subplot
        ax.legend([bp_real["boxes"][0], bp_generated["boxes"][0]], ['Real', 'Generated'], loc='lower left', fontsize=10, frameon=False)

    plt.xlabel('Food Categories', fontsize=16, fontweight='bold')
    fig.suptitle(f'Aggregated {metric} by Meal Type and Food Category', fontsize=20, fontweight='bold')
    plt.tight_layout()
    plt.subplots_adjust(top=0.9, bottom=0.1)
    plt.savefig(f'tests/{metric}.png', dpi=300)
    
def plot_metric_box_and_whisker(data, metric_cols, corrected_pvals_dict):
    meal_types = ['breakfast', 'lunch', 'dinner']
    
    # Adjusting figure size for readability
    fig, axes = plt.subplots(nrows=len(meal_types), ncols=1, figsize=(15, len(metric_cols) * 2), sharex=True)

    for ax_idx, meal_type in enumerate(meal_types):
        ax = axes[ax_idx]
        # Calculating positions for real and generated data points
        positions_real = np.arange(len(metric_cols)) * 2 - 0.2
        positions_generated = np.arange(len(metric_cols)) * 2 + 0.2

        # Extracting real and generated data for each metric
        real_data = [data['Real'][metric][meal_type] if metric in data['Real'] and meal_type in data['Real'][metric] else [] for metric in metric_cols]
        generated_data = [data['Generated'][metric][meal_type] if metric in data['Generated'] and meal_type in data['Generated'][metric] else [] for metric in metric_cols]

        # Plotting real and generated data
        bp_real = ax.boxplot(real_data, positions=positions_real, widths=0.4, patch_artist=True, boxprops=dict(facecolor="red"), showfliers=False)
        bp_generated = ax.boxplot(generated_data, positions=positions_generated, widths=0.4, patch_artist=True, boxprops=dict(facecolor="blue"), showfliers=False)

        # Add statistical significance testing annotations using corrected p-values
        for i, metric in enumerate(metric_cols):
            if meal_type in corrected_pvals_dict[metric]:
                corrected_p_value = corrected_pvals_dict[metric][meal_type][i]  # Adjust indexing as needed
                # Calculate the maximum y value for plotting p-value annotations
                max_y_value = max([max(data) if data else 0 for data in (real_data[i], generated_data[i])]) + 0.1
                add_pvalue_bracket(ax, positions_real[i], positions_generated[i], max_y_value, 0.05, corrected_p_value)

        ax.set_ylabel('Metric Scores')
        ax.set_title(f'{meal_type.capitalize()} Meal Metrics', loc='left')
        ax.set_xticks(np.arange(len(metric_cols)) * 2)
        ax.set_xticklabels(metric_cols, rotation=45, ha="right")
        ax.grid(True)

    plt.xlabel('Metrics')
    fig.suptitle('Aggregated Metrics by Meal Type')
    plt.tight_layout()
    plt.subplots_adjust(top=0.95, bottom=0.1)
    plt.legend([bp_real["boxes"][0], bp_generated["boxes"][0]], ['Real', 'Generated'], loc='upper right')
    plt.savefig(f'tests/aggregated_metrics.png', dpi=1500)

def generate_food_combination(data, meal_to_generate, meal_type):
    presence_model = PresenceModel(data.shape[1], meal_type, 'config.json')
    decoder_model = PresenceModel.load_model(presence_model.model_save_path + 'optimal/decoder_model.keras')
    random_latent_vectors = np.random.normal(size=(meal_to_generate, presence_model.config["presence_model"]['latent_dim']))
    generated_meals = decoder_model.predict(random_latent_vectors, verbose = 0)
    generated_meals = (generated_meals > 0.5).astype(int)
    return generated_meals

def generate_food_portions(generated_meal_combinations, meal_type):
    portion_model = PortionModel(generated_meal_combinations.shape[1], meal_type, 'config.json')
    portion_model = PortionModel.load_model(portion_model.model_save_path + 'optimal/model.keras')
    generated_meal_portions = portion_model.predict(generated_meal_combinations, verbose = 0)    
    return generated_meal_portions

def log_meals(data, generated_meal_combinations, generated_meal_portions, meal_type, log):
    food_codes = data.columns.astype(int)
    food_mapping = load_food_mapping('data/mappings/food_ingredient_map.pkl')

    if len(generated_meal_combinations.shape) == 1:
        generated_meal_combinations = [generated_meal_combinations]
        generated_meal_portions = [generated_meal_portions]

    for i, (meal, portions) in enumerate(zip(generated_meal_combinations, generated_meal_portions)):
        log.info(f"Generated {meal_type} meal {i+1}:")
        for j, present in enumerate(meal):
            if present == 1:
                food_item = food_mapping.get(food_codes[j], "Unknown Food")
                portion_size = portions[j]
                log.info(f"\t{food_item}: {portion_size:.2f}g")

def check_nutritional_data_availability(cache_dir, logger):
    food_to_nut_map_dir = 'data/mappings/food_to_nutrient.pkl'
    os.makedirs(cache_dir, exist_ok=True)
    if not os.path.exists(food_to_nut_map_dir):
        aggregate_nutritional_data(cache_dir, logger)
    else:
        logger.info(f"{food_to_nut_map_dir} already exists, using cached data")

def aggregate_nutritional_data(cache_dir, log):
    nutrient_urls = [
        "https://www.ars.usda.gov/ARSUserFiles/80400530/apps/2015-2016%20FNDDS%20At%20A%20Glance%20-%20FNDDS%20Nutrient%20Values.xlsx", # 2015-2016
        "https://www.ars.usda.gov/ARSUserFiles/80400530/apps/2017-2018%20FNDDS%20At%20A%20Glance%20-%20FNDDS%20Nutrient%20Values.xlsx", # 2017-2018
        "https://www.ars.usda.gov/ARSUserFiles/80400530/apps/2019-2020%20FNDDS%20At%20A%20Glance%20-%20FNDDS%20Nutrient%20Values.xlsx"  # 2018-2019
    ]
    nutrient_urls.sort(key=extract_year_from_filename, reverse=True)
    nutrient_data = pd.DataFrame(columns=['Food code'])
    for file_url in nutrient_urls:
        data = fetch_data(file_url, cache_dir, log)
        data.columns = [re.sub(r'\s*(\()', r'\n\1', col) for col in data.columns]
        new_codes = ~data['Food code'].isin(nutrient_data['Food code'])
        nutrient_data = pd.concat([nutrient_data, data[new_codes]], ignore_index=True)
    to_pickle(nutrient_data, 'data/mappings/food_to_nutrient.pkl')

def is_novel_combination(meal_combination, existing_meals_df):
    existing_meals_binary = (existing_meals_df > 0).astype(int)
    return not any((existing_meals_binary == meal_combination).all(axis=1))

def generate_meals(meal_type, num_meals_to_generate, save, log):
    data = pd.read_pickle(f'data/processed/real_meals/dataframes/{meal_type}_sparsity_filtered.pkl')
    food_mapping = load_food_mapping('data/mappings/food_ingredient_map.pkl')

    generated_meal_combinations, generated_meal_portions = [], []
    
    if log is not None:
        log.info(f"="*10 + f" Generating {num_meals_to_generate} {meal_type} meals " + f"="*10)

    while len(generated_meal_combinations) < num_meals_to_generate:
        batch_size = num_meals_to_generate - len(generated_meal_combinations)
        meal_combinations_batch = generate_food_combination(data, batch_size, meal_type)
        meal_portions_batch = generate_food_portions(meal_combinations_batch, meal_type)

        if save: # For generation
            unique_meals = set()
            for i in range(batch_size):
                single_meal_combination = meal_combinations_batch[i]
                single_meal_portion = meal_portions_batch[i]

                meal_tuple = tuple(single_meal_combination)
                selected_food_portions = single_meal_portion[single_meal_combination == 1]
                all_codes_exist = all(food_mapping.get(data.columns[j]) for j in range(len(single_meal_combination)) if single_meal_combination[j] == 1)

                if meal_tuple not in unique_meals and is_novel_combination(single_meal_combination, data) and (selected_food_portions > 0).all() and all_codes_exist and len(selected_food_portions) > 1:
                    unique_meals.add(meal_tuple)
                    generated_meal_combinations.append(single_meal_combination)
                    generated_meal_portions.append(single_meal_portion)
                        
        else: # For evaluation
            for i in range(batch_size):
                generated_meal_combinations.append(meal_combinations_batch[i])
                generated_meal_portions.append(meal_portions_batch[i])
            
    generated_meal_combinations = np.array(generated_meal_combinations).squeeze()
    generated_meal_portions = np.array(generated_meal_portions).squeeze()
    if log is not None:
        log_meals(data, generated_meal_combinations, generated_meal_portions, meal_type, log)
    
    generated_meals_df = pd.DataFrame(generated_meal_combinations, columns=data.columns)
    generated_meals_df = generated_meals_df * np.array(generated_meal_portions)
    generated_meals_df.index = [f'{meal_type}_{i+1}' for i in range(len(generated_meals_df))]
    
    if save:
        output_dir = f'data/processed/generated_meals/dataframes'
        os.makedirs(output_dir, exist_ok=True)
        to_pickle(generated_meals_df, f'{output_dir}/{meal_type}_meals.pkl')
    else:
        return generated_meals_df

def evaluate_and_visualize_embeddings(real_embeddings, generated_embeddings):
    # Ensure 'node' or meal names are columns and not indices
    if real_embeddings.index.name == 'node':
        real_embeddings = real_embeddings.reset_index()
    if generated_embeddings.index.name == 'node':
        generated_embeddings = generated_embeddings.reset_index()

    # Standardize features of real embeddings
    scaler = MinMaxScaler()
    real_scaled_embeddings = scaler.fit_transform(real_embeddings.iloc[:, 1:])

    # Heuristic for DBSCAN's eps
    min_samples = 2 * real_scaled_embeddings.shape[1]
    nearest_neighbors = NearestNeighbors(n_neighbors=min_samples)
    nearest_neighbors.fit(real_scaled_embeddings)
    distances, indices = nearest_neighbors.kneighbors(real_scaled_embeddings)

    min_samples = int(min_samples/3)
    # DBSCAN clustering on real meals
    # Assume `real_scaled_embeddings` are your real meal embeddings
    dbscan = DBSCAN(eps=0.25, min_samples=min_samples)  # Adjust these parameters as needed
    real_clusters = dbscan.fit_predict(real_scaled_embeddings)
    
    # Scale generated meal embeddings
    generated_scaled_embeddings = scaler.transform(generated_embeddings.iloc[:, 1:])

    # Calculate practicality scores
    practicality_scores = {}
    impractical_generated = []  # List to hold impractical meals

    for index, gen_emb in enumerate(generated_scaled_embeddings):
        gen_label = generated_embeddings.iloc[index, 0]

        # Find the nearest real meal cluster
        nearest_real_meal_index = np.argmin(pairwise_distances([gen_emb], real_scaled_embeddings))
        gen_cluster = real_clusters[nearest_real_meal_index]

        # Check if the nearest real meal is an outlier
        if gen_cluster == -1:
            impractical_generated.append(gen_label)
            continue

        # Calculate distances to all real meals in the same cluster
        cluster_indices = np.where(real_clusters == gen_cluster)[0]
        distances = pairwise_distances([gen_emb], real_scaled_embeddings[cluster_indices])
        practicality_scores[gen_label] = np.median(distances)

    # Identify practical generated meals based on threshold
    threshold = np.percentile(list(practicality_scores.values()), 95)
    practical_generated = [label for label, score in practicality_scores.items() if score <= threshold]

    # Add any remaining generated meals not already classified as impractical
    for label in generated_embeddings.iloc[:, 0]:
        if label not in practical_generated and label not in impractical_generated:
            impractical_generated.append(label)
            
    return practical_generated, impractical_generated

def practicality_filtration(generated_embedding_file):
    real_embeddings = pd.read_pickle('results/embedding_model/real_meal_embeddings.pkl')
    generated_embeddings = pd.read_pickle(generated_embedding_file)

    real_meals = real_embeddings[real_embeddings['node'].str.contains('real')]
    generated_meals = generated_embeddings[generated_embeddings['node'].str.contains('generated')]
    real_meals = generated_embeddings[generated_embeddings['node'].str.contains('real')]
    practical_generated, impractical_generated = evaluate_and_visualize_embeddings(real_meals, generated_meals)
    print(f'Practical meals: {len(practical_generated)}')
    print(f'Impractical meals: {len(impractical_generated)}')
    return practical_generated

def apply_serving_sizes(meal_df, category_mapping, serving_size_mapping):
    # Create a new DataFrame to hold the adjusted meal data
    adjusted_meal_df = meal_df.copy()

    for food_code in adjusted_meal_df.columns:
        # Find the category of the current food item
        category = category_mapping.get(food_code, 'Other')  # Default to 'Other' if not found

        # Get the recommended serving size for this category
        serving_size = serving_size_mapping.get(category, 28)  # Default to 28g if not found

        # Replace non-zero portion sizes with the recommended serving size
        adjusted_meal_df[food_code] = adjusted_meal_df[food_code].apply(lambda portion: serving_size if portion > 0 else 0)

    return adjusted_meal_df

def bootstrap_meals(metric_cols, num_bootstrap, num_days, meal_types, log, generate_meals_func=None):
    # Initialization of random seeds and data loading
    seeds = [random.randint(0, 9999) for _ in range(num_bootstrap)]
    real_meals_dict = {meal_type: pd.read_pickle(f'data/processed/real_meals/dataframes/{meal_type}_sparsity_filtered.pkl') for meal_type in meal_types}
    mapping_df = pd.read_pickle('data/mappings/food_ingredient_map.pkl')
    food_category_mapping = dict(zip(mapping_df['Food code'], mapping_df['WWEIA Main Category description']))
    unique_food_categories = set(food_category_mapping.values())

    # Initialize metrics dictionaries
    all_metrics = {metric: {meal_type: [] for meal_type in meal_types} for metric in metric_cols}
    serving_size_all_metrics = {metric: {meal_type: [] for meal_type in meal_types} for metric in metric_cols}
    category_deviations = {meal_type: {category: [] for category in unique_food_categories} for meal_type in meal_types}
    category_balance = {meal_type: {category: [] for category in unique_food_categories} for meal_type in meal_types}
    macronutrient_coverage = {nutrient: [] for nutrient in MACRONUTRIENTS}
    micronutrient_coverage = {nutrient: [] for nutrient in MICRONUTRIENTS}

    # Main bootstrapping loop
    with tqdm(total=num_bootstrap, desc="Bootstrapping meals") as pbar:
        for seed in seeds:
            bootstrap_sample = {meal_type: pd.DataFrame() for meal_type in meal_types}
            generated_meals = {meal_type: [] for meal_type in meal_types}
            
            for meal_type in meal_types:
                if generate_meals_func:
                    meal_df = generate_meals_func(meal_type, num_days, save=False, log=None)
                    to_pickle(meal_df, f'data/processed/generated_meals/dataframes/{meal_type}_meals.pkl')
                    generated_meals[meal_type].append(meal_df)
                else:
                    meal_df = real_meals_dict[meal_type].sample(n=num_days, replace=True, random_state=seed)
                    meal_df.index = [f'{meal_type}_{i+1}' for i in range(len(meal_df))]
                    bootstrap_sample[meal_type] = meal_df

            if generate_meals_func:
                # Placeholder for graph creation, embedding training, and filtration
                # Assume these are utility functions defined elsewhere
                print('Creating Meal Graph')
                create_meal_graph('path/to/graph_file', 'generated', None)
                print('Training Embedding Model')
                train_embedding_model('path/to/graph_file', 'path/to/embedding_file', None, None)
                print('Practicality Filtration')
                practical_meals = practicality_filtration('path/to/embedding_file')
                
                meal_graph = pd.read_csv('path/to/graph_file', header=None, names=['nodeA', 'nodeB', 'grams'])
                for meal_type, meal_list in generated_meals.items():
                    combined_meals_df = pd.concat(meal_list, ignore_index=True)
                    filtered_graph = meal_graph[
                        (meal_graph['nodeA'].isin(practical_meals)) &
                        (meal_graph['nodeA'].apply(lambda x: meal_type in x)) &
                        (meal_graph['nodeB'].str.startswith('food_'))
                    ]
                    if not filtered_graph.empty:
                        meal_df = filtered_graph.pivot_table(index='nodeA', columns='nodeB', values='grams', aggfunc='sum').fillna(0)
                        meal_df.columns = meal_df.columns.str.replace('food_', '').astype(int)
                    else:
                        meal_df = pd.DataFrame()
                    meal_df.index = [f'{meal_type}_{i+1}' for i in range(len(meal_df))]
                    bootstrap_sample[meal_type] = meal_df
            
            pbar.update(1)
    
    # Save the generated metrics to disk
    os.makedirs('results/evaluation/', exist_ok=True)
    fn = "generated" if generate_meals_func else "real"
    to_pickle(all_metrics, f'results/evaluation/{fn}_meal_metrics_{num_bootstrap}_{num_days}.pkl')
    to_pickle(serving_size_all_metrics, f'results/evaluation/{fn}_serving_size_meal_metrics_{num_bootstrap}_{num_days}.pkl')

    return all_metrics, serving_size_all_metrics


def bootstrap_meals(metric_cols, num_bootstrap, num_days, meal_types, log, generate_meals_func=None):
    all_metrics = {metric: {meal_type: [] for meal_type in meal_types} for metric in metric_cols}
    serving_size_all_metrics = {metric: {meal_type: [] for meal_type in meal_types} for metric in metric_cols}
    seeds = [random.randint(0, 9999) for _ in range(num_bootstrap)]
    real_meals_dict = {meal_type: pd.read_pickle(f'data/processed/real_meals/dataframes/{meal_type}_sparsity_filtered.pkl') for meal_type in meal_types}

    mapping_df = pd.read_pickle('data/mappings/food_ingredient_map.pkl')
    food_category_mapping = dict(zip(mapping_df['Food code'], mapping_df['WWEIA Main Category description']))
    unique_food_categories = set(val for val in food_category_mapping.values())

    graph_file = 'tests/temp_graph.csv'
    temp_embedding_file = 'tests/temp_embeddings.pkl'

    with tqdm(total=num_bootstrap, desc="Bootstrapping meals") as pbar:
        for seed in seeds:
            bootstrap_sample = {meal_type: pd.DataFrame() for meal_type in meal_types}                
            generated_meals = {}
            for meal_type in meal_types:
                generated_meals[meal_type] = []
                if generate_meals_func:
                    # Generate or collect meals for each meal type without saving or filtering yet
                    meal_df = generate_meals_func(meal_type, num_days, save=False, log=None)
                    to_pickle(meal_df, f'data/processed/generated_meals/dataframes/{meal_type}_meals.pkl')
                    generated_meals[meal_type].append(meal_df)
                else:
                    # Sample real meals if no generation function is provided
                    meal_df = real_meals_dict[meal_type].sample(n=num_days, replace=True, random_state=seed)
                    meal_df.index = [f'{meal_type}_{i+1}' for i in range(len(meal_df))]
                    bootstrap_sample[meal_type] = meal_df
                    
            if generate_meals_func:
                # After collecting all meals, proceed with unified graph creation, embedding training, and filtration
                print('Creating Meal Graph')
                create_meal_graph(graph_file, 'generated', None)
                print('Training Embedding Model')
                train_embedding_model(graph_file, temp_embedding_file, None, None)
                print('Practicality Filtration')
                practical_meals = practicality_filtration(temp_embedding_file)

                meal_graph = pd.read_csv(graph_file, header=None, names=['nodeA', 'nodeB', 'grams'])
                for meal_type, meal_list in generated_meals.items():
                    # Combine all meal dataframes for the current meal type into a single dataframe
                    combined_meals_df = pd.concat(meal_list, ignore_index=True)
                    
                    # Filter meal graph for the current meal type and practical meals
                    filtered_graph = meal_graph[(meal_graph['nodeA'].isin(practical_meals)) &
                                                (meal_graph['nodeA'].apply(lambda x: meal_type in x)) &
                                                (meal_graph['nodeB'].str.startswith('food_'))]

                    if not filtered_graph.empty:
                        # Pivot and fill missing values if there are any practical meals of the current type
                        meal_df = filtered_graph.pivot_table(index='nodeA', columns='nodeB', values='grams', aggfunc='sum').fillna(0)
                        meal_df.columns = meal_df.columns.str.replace('food_', '').astype(int)
                    else:
                        # Handle case where no practical meals are left after filtration
                        meal_df = pd.DataFrame()

                    # Update the index to reflect the original meal identifiers
                    meal_df.index = [f'{meal_type}_{i+1}' for i in range(len(meal_df))]
                    bootstrap_sample[meal_type] = meal_df
                
            print('Calculating Meal Metrics')
            calculated_metrics, plot_metrics = calculate_meal_metrics(bootstrap_sample, real_meals_dict, food_category_mapping, num_days)
            
            print('Calculating Serving Size Meal Metrics')
            mapping_df = pd.read_pickle('data/mappings/food_ingredient_map.pkl')
            # Prepare the category mapping (food code to category description)
            category_mapping = mapping_df.set_index('Food code')['WWEIA Main Category description'].to_dict()
            # Apply the serving sizes to each meal DataFrame
            bootstrap_sample_cpy = bootstrap_sample.copy()
            for meal_type, meal_df in bootstrap_sample_cpy.items():
                bootstrap_sample_cpy[meal_type] = apply_serving_sizes(meal_df, category_mapping, usda_recommended_serving_size_mapping)
            serving_size_calculated_metrics, serving_size_plot_metrics = calculate_meal_metrics(bootstrap_sample_cpy, real_meals_dict, food_category_mapping, num_days)
            if generate_meals_func is None:
                fn = 'real'
            else:
                fn = 'generated'
                
            to_pickle(serving_size_calculated_metrics, f'tests/serving_size_{fn}_{num_bootstrap}_{num_days}_calculated_metrics.pkl')
            to_pickle(serving_size_plot_metrics, f'tests/serving_size_{fn}_{num_bootstrap}_{num_days}_plot_metrics.pkl')
            
            category_deviations = {meal_type: {category: [] for category in unique_food_categories} for meal_type in meal_types}
            category_balance = {meal_type: {category: [] for category in unique_food_categories} for meal_type in meal_types}
            macronutrient_coverage = {nutrient: [] for nutrient in MACRONUTRIENTS}
            micronutrient_coverage = {nutrient: [] for nutrient in MICRONUTRIENTS}
            
            assert plot_metrics is not serving_size_plot_metrics
            
            for meal_type in meal_types:
                for category in unique_food_categories:
                    if category in plot_metrics['Average Category Deviations'][meal_type].keys():
                        avg_deviation = plot_metrics['Average Category Deviations'][meal_type][category]
                        category_deviations[meal_type][category].append(avg_deviation)
                    if category in plot_metrics['Average Category Balance'][meal_type].keys():
                        avg_category_balance = plot_metrics['Average Category Balance'][meal_type][category]
                        category_balance[meal_type][category].append(avg_category_balance)
                for nutrient in MACRONUTRIENTS:
                    if nutrient in plot_metrics['Average Macronutrient Coverage'].keys():
                        avg_macronutrient_coverage = plot_metrics['Average Macronutrient Coverage'][nutrient]
                        macronutrient_coverage[nutrient].append(avg_macronutrient_coverage)
                for nutrient in MICRONUTRIENTS:
                    if nutrient in plot_metrics['Average Micronutrient Coverage'].keys():
                        avg_micronutrient_coverage = plot_metrics['Average Micronutrient Coverage'][nutrient]
                        micronutrient_coverage[nutrient].append(avg_micronutrient_coverage)                        
            for metric in calculated_metrics:
                for meal_type in meal_types:
                    if metric == "Micronutrient Coverage" or metric == "Macronutrient Coverage":
                        all_metrics[metric][meal_type].append(calculated_metrics[metric])
                    else:
                        all_metrics[metric][meal_type].append(calculated_metrics[metric][meal_type])
            all_metrics['Average Category Balance'] = category_balance
            all_metrics['Average Category Deviations'] = category_deviations
            all_metrics['Average Macronutrient Coverage'] = macronutrient_coverage
            all_metrics['Average Micronutrient Coverage'] = micronutrient_coverage
        
            category_deviations = {meal_type: {category: [] for category in unique_food_categories} for meal_type in meal_types}
            category_balance = {meal_type: {category: [] for category in unique_food_categories} for meal_type in meal_types}
            macronutrient_coverage = {nutrient: [] for nutrient in MACRONUTRIENTS}
            micronutrient_coverage = {nutrient: [] for nutrient in MICRONUTRIENTS}
                    
            for meal_type in meal_types:
                for category in unique_food_categories:
                    if category in serving_size_plot_metrics['Average Category Deviations'][meal_type].keys():
                        avg_deviation = serving_size_plot_metrics['Average Category Deviations'][meal_type][category]
                        category_deviations[meal_type][category].append(avg_deviation)
                    if category in serving_size_plot_metrics['Average Category Balance'][meal_type].keys():
                        avg_category_balance = serving_size_plot_metrics['Average Category Balance'][meal_type][category]
                        category_balance[meal_type][category].append(avg_category_balance)
                for nutrient in MACRONUTRIENTS:
                    if nutrient in serving_size_plot_metrics['Average Macronutrient Coverage'].keys():
                        avg_macronutrient_coverage = serving_size_plot_metrics['Average Macronutrient Coverage'][nutrient]
                        macronutrient_coverage[nutrient].append(avg_macronutrient_coverage)
                for nutrient in MICRONUTRIENTS:
                    if nutrient in serving_size_plot_metrics['Average Micronutrient Coverage'].keys():
                        avg_micronutrient_coverage = serving_size_plot_metrics['Average Micronutrient Coverage'][nutrient]
                        micronutrient_coverage[nutrient].append(avg_micronutrient_coverage)                        
            for metric in serving_size_calculated_metrics:
                for meal_type in meal_types:
                    if metric == "Micronutrient Coverage" or metric == "Macronutrient Coverage":
                        serving_size_all_metrics[metric][meal_type].append(serving_size_calculated_metrics[metric])
                    else:
                        serving_size_all_metrics[metric][meal_type].append(serving_size_calculated_metrics[metric][meal_type])
            serving_size_all_metrics['Average Category Balance'] = category_balance
            serving_size_all_metrics['Average Category Deviations'] = category_deviations
            serving_size_all_metrics['Average Macronutrient Coverage'] = macronutrient_coverage
            serving_size_all_metrics['Average Micronutrient Coverage'] = micronutrient_coverage
            pbar.update(1)
            
    os.makedirs('results/evaluation/', exist_ok=True)
    if generate_meals_func == None:
        fn = "real"
    else: 
        fn = "generated" 
    to_pickle(all_metrics, f'results/evaluation/{fn}_meal_metrics_{num_bootstrap}_{num_days}.pkl')
    to_pickle(serving_size_all_metrics, f'results/evaluation/{fn}_serving_size_meal_metrics_{num_bootstrap}_{num_days}.pkl')
    return all_metrics, serving_size_all_metrics


def permutation_test(sample1, sample2, n_permutations=10000):
    diff_obs = np.abs(np.mean(sample1) - np.mean(sample2))
    pooled = np.hstack([sample1, sample2])
    diff_perms = []
    for _ in range(n_permutations):
        np.random.shuffle(pooled)
        diff_perm = np.abs(np.mean(pooled[:len(sample1)]) - np.mean(pooled[len(sample1):]))
        diff_perms.append(diff_perm)
    p_value = np.sum(diff_perms >= diff_obs) / n_permutations
    return p_value

def collect_p_values(real_meals_metrics, generated_meals_metrics, metric_cols, use_permutation=False, n_permutations=10000):
    p_values = []
    p_value_mappings = []

    for metric in metric_cols:
        if metric in ['Average Category Deviations', 'Average Category Balance']:
            for meal_type, categories_data in real_meals_metrics[metric].items():
                for category, real_data in categories_data.items():
                    generated_data = generated_meals_metrics[metric][meal_type][category]
                    
                    if real_data and generated_data:
                        if use_permutation:
                            p_value = permutation_test(np.array(real_data), np.array(generated_data), n_permutations)
                        else:
                            _, p_value = stats.mannwhitneyu(real_data, generated_data, alternative='two-sided')
                        p_values.append(p_value)
                        p_value_mappings.append((metric, meal_type, category))
    
    return p_values, p_value_mappings

def map_corrected_pvals_to_comparisons(corrected_pvals, p_value_mappings):
    corrected_pvals_dict = {}
    for (metric, meal_type, category), corrected_pval in zip(p_value_mappings, corrected_pvals):
        if metric not in corrected_pvals_dict:
            corrected_pvals_dict[metric] = {}
        if meal_type not in corrected_pvals_dict[metric]:
            corrected_pvals_dict[metric][meal_type] = {}
        corrected_pvals_dict[metric][meal_type][category] = corrected_pval
    return corrected_pvals_dict

def plot_aggregated_metrics(real_meals_metrics, generated_meals_metrics, metric_cols):
    p_values, p_value_mappings = collect_p_values(real_meals_metrics, generated_meals_metrics, metric_cols)
    _, corrected_pvals, _, _ = multipletests(p_values, alpha=0.05, method='fdr_bh')
    corrected_pvals_dict = map_corrected_pvals_to_comparisons(corrected_pvals, p_value_mappings)

    for metric in metric_cols:
        data = {
            'Real': real_meals_metrics[metric],
            'Generated': generated_meals_metrics[metric]
        }
        if metric == 'Average Category Deviations' or metric == 'Average Category Balance':
            plot_category_box_and_whisker(data, metric, corrected_pvals_dict[metric])
        wanted_cols = [x for x in metric_cols if x not in ["Average Category Deviations", "Average Category Balance", "Average Macronutrient Coverage", "Average Micronutrient Coverage"]]
    data = {
        'Real': {k: real_meals_metrics[k] for k in wanted_cols},
        'Generated': {k: generated_meals_metrics[k] for k in wanted_cols}
    }
    #plot_metric_box_and_whisker(data, wanted_cols, corrected_pvals_dict)

def format_metrics(metric_values):
    print("Formatting metrics for:", metric_values)
    if isinstance(metric_values, dict):
        return 0
    
    mean = np.mean(metric_values)
    std = np.std(metric_values)
    return f"{mean:.2f} ± {std:.2f}"

def create_metrics_table(real_meals_metrics, generated_meals_metrics, serving_size_real_meals_metrics, serving_size_generated_meals_metrics, metric_cols, meal_types):
    table = PrettyTable()
    header = ['Metric', 'Meal Type', 'Real', 'Generated', 'Serving Size Real', 'Serving Size Generated']
    table.field_names = header

    for metric in metric_cols:
        if metric not in ["Average Category Deviations", "Average Category Balance", "Average Macronutrient Coverage", "Average Micronutrient Coverage"]:
            for meal_type in meal_types:
                real_metrics = real_meals_metrics[metric][meal_type]
                formatted_real_metrics = format_metrics(real_metrics)
                generated_metrics = generated_meals_metrics[metric][meal_type]
                formatted_generated_metrics = format_metrics(generated_metrics)
                serving_size_generated_metrics = serving_size_generated_meals_metrics[metric][meal_type]
                formatted_serving_size_generated_metrics = format_metrics(serving_size_generated_metrics)
                serving_size_real_metrics = serving_size_real_meals_metrics[metric][meal_type]
                formatted_serving_size_real_metrics = format_metrics(serving_size_real_metrics)
                table.add_row([metric, meal_type, formatted_real_metrics, formatted_generated_metrics, formatted_serving_size_real_metrics, formatted_serving_size_generated_metrics])
    
    return table

def calculate_averages_for_plotting(meals_metrics, metric_cols, meal_types):
    averages = {'Real': {}, 'Generated': {},  'Serving Size Real': {},  'Serving Size Generated': {}}
    # Initialize structure with meal types as keys
    for provider in ['Real', 'Generated', 'Serving Size Real', 'Serving Size Generated']:
        for meal_type in meal_types:
            averages[provider][meal_type] = {}
    
    for metric in metric_cols:
        for meal_type in meal_types:
            # Initialize dictionary for nested metrics
            if metric not in averages['Real'][meal_type]:
                averages['Real'][meal_type][metric] = {}
            if metric not in averages['Generated'][meal_type]:
                averages['Generated'][meal_type][metric] = {}
            if metric not in averages['Serving Size Real'][meal_type]:
                averages['Serving Size Real'][meal_type][metric] = {}
            if metric not in averages['Serving Size Generated'][meal_type]:
                averages['Serving Size Generated'][meal_type][metric] = {}
                
            # Check if the metric is a nested field
            if metric in ["Average Category Deviations", "Average Category Balance"]:
                if meals_metrics['Real'][metric][meal_type]:
                    for nutrient, values in meals_metrics['Real'][metric][meal_type].items():
                        # Calculate the mean for each nutrient and store it
                        real_avg = np.mean(values) if values else 0
                        averages['Real'][meal_type][metric][nutrient] = real_avg
                if meals_metrics['Serving Size Real'][metric][meal_type]:
                    for nutrient, values in meals_metrics['Serving Size Real'][metric][meal_type].items():
                        # Calculate the mean for each nutrient and store it
                        serving_size_real_avg = np.mean(values) if values else 0
                        averages['Serving Size Real'][meal_type][metric][nutrient] = serving_size_real_avg
                if meals_metrics['Generated'][metric][meal_type]:
                    for nutrient, values in meals_metrics['Generated'][metric][meal_type].items():
                        # Calculate the mean for each nutrient and store it
                        generated_avg = np.mean(values) if values else 0
                        averages['Generated'][meal_type][metric][nutrient] = generated_avg
                if meals_metrics['Serving Size Generated'][metric][meal_type]:
                    for nutrient, values in meals_metrics['Serving Size Generated'][metric][meal_type].items():
                        # Calculate the mean for each nutrient and store it
                        serving_size_generated_avg = np.mean(values) if values else 0
                        averages['Serving Size Generated'][meal_type][metric][nutrient] = serving_size_generated_avg
            elif metric in ["Average Macronutrient Coverage", "Average Micronutrient Coverage"]:
                if meals_metrics['Real'][metric]:
                    for nutrient, values in meals_metrics['Real'][metric].items():
                        # Calculate the mean for each nutrient and store it
                        real_avg = np.mean(values) if values else 0
                        averages['Real'][meal_type][metric][nutrient] = real_avg
                if meals_metrics['Serving Size Real'][metric]:
                    for nutrient, values in meals_metrics['Serving Size Real'][metric].items():
                        # Calculate the mean for each nutrient and store it
                        serving_size_real_avg = np.mean(values) if values else 0
                        averages['Serving Size Real'][meal_type][metric][nutrient] = serving_size_real_avg
                if meals_metrics['Generated'][metric]:
                    for nutrient, values in meals_metrics['Generated'][metric].items():
                        # Calculate the mean for each nutrient and store it
                        generated_avg = np.mean(values) if values else 0
                        averages['Generated'][meal_type][metric][nutrient] = generated_avg
                if meals_metrics['Serving Size Generated'][metric]:
                    for nutrient, values in meals_metrics['Serving Size Generated'][metric].items():
                        # Calculate the mean for each nutrient and store it
                        serving_size_generated_avg = np.mean(values) if values else 0
                        averages['Serving Size Generated'][meal_type][metric][nutrient] = serving_size_generated_avg
            else:
                # For non-nested metrics, just calculate the overall average
                real_avg = np.mean(meals_metrics['Real'][metric][meal_type])
                generated_avg = np.mean(meals_metrics['Generated'][metric][meal_type])
                serving_size_real_avg = np.mean(meals_metrics['Serving Size Real'][metric][meal_type])
                serving_size_generated_avg = np.mean(meals_metrics['Serving Size Generated'][metric][meal_type])
                averages['Real'][meal_type][metric] = real_avg
                averages['Generated'][meal_type][metric] = generated_avg
                averages['Serving Size Real'][meal_type][metric] = serving_size_real_avg
                averages['Serving Size Generated'][meal_type][metric] = serving_size_generated_avg

    return averages

def plot_spider_chart(averages, metric_cols):
    num_vars = len(metric_cols)
    angles = np.linspace(0, 2*np.pi, num_vars, endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))
    values_real = [np.mean(averages['Real'][metric]) for metric in metric_cols]
    values_real += values_real[:1]  # Ensuring the list is cyclic for a closed plot
    ax.plot(angles, values_real, linewidth=1, linestyle='solid', label='Real')
    ax.fill(angles, values_real, 'b', alpha=0.1)

    values_generated = [np.mean(averages['Generated'][metric]) for metric in metric_cols]
    values_generated += values_generated[:1]
    ax.plot(angles, values_generated, linewidth=1, linestyle='solid', label='Generated')
    ax.fill(angles, values_generated, 'r', alpha=0.1)

    ax.set_thetagrids(np.degrees(angles[:-1]), metric_cols)
    plt.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))
    plt.savefig('tests/average_spider_plot.png', dpi=1500)
    
def plot_spider_chart_per_meal_type(averages, general_metrics, nutritional_metrics, meal_types):
    num_rows = len(meal_types)
    num_cols = 1 + len(nutritional_metrics)  # 1 for general metrics + number of nutritional categories
    fig, axs = plt.subplots(num_rows, num_cols, figsize=(num_cols * 6, num_rows * 6), subplot_kw=dict(polar=True))

    if num_rows == 1:  # Adjust if only one meal type for consistent indexing
        axs = np.array([axs])

    for i, meal_type in enumerate(meal_types):
        values_real = [averages['Real'][meal_type].get(metric, 0) for metric in general_metrics]
        values_generated = [averages['Generated'][meal_type].get(metric, 0) for metric in general_metrics]
        values_serving_size_real = [averages['Serving Size Real'][meal_type].get(metric, 0) for metric in general_metrics]
        values_serving_size_generated = [averages['Serving Size Generated'][meal_type].get(metric, 0) for metric in general_metrics]
        plot_spider_subchart(axs[i, 0], values_real, values_generated, values_serving_size_real, values_serving_size_generated,  general_metrics, meal_type + " - General Metrics")

        for j, nutrient_category in enumerate(nutritional_metrics, start=1):
            if nutrient_category in averages['Real'][meal_type]:
                nutrient_labels = list(averages['Real'][meal_type][nutrient_category].keys())
                values_real = [averages['Real'][meal_type][nutrient_category].get(nutrient, 0) for nutrient in nutrient_labels]
                values_generated = [averages['Generated'][meal_type][nutrient_category].get(nutrient, 0) for nutrient in nutrient_labels]
                values_serving_size_real = [averages['Serving Size Real'][meal_type][nutrient_category].get(nutrient, 0) for nutrient in nutrient_labels]
                values_serving_size_generated = [averages['Serving Size Generated'][meal_type][nutrient_category].get(nutrient, 0) for nutrient in nutrient_labels]
                plot_spider_subchart(axs[i, j], values_real, values_generated, values_serving_size_real, values_serving_size_generated, nutrient_labels, meal_type + f" - {nutrient_category}")
    
    plt.tight_layout()
    plt.savefig('tests/final_spider_plot.png', dpi=1500)
    
def plot_averages_across_meal_types(averages, general_metrics, nutritional_metrics, meal_types):
    num_cols = 1 + len(nutritional_metrics)  # 1 for general metrics + number of nutritional categories
    fig, axs = plt.subplots(1, num_cols, figsize=(num_cols * 6, 6), subplot_kw=dict(polar=True))

    # Initialize max_value for normalization
    max_value = 0

    # Update max_value based on general_metrics
    for metric in general_metrics:
        for meal_type in meal_types:
            value = averages['Real'][meal_type].get(metric, 0)
            max_value = max(max_value, abs(value))
            value = averages['Generated'][meal_type].get(metric, 0)
            max_value = max(max_value, abs(value))
            value = averages['Serving Size Real'][meal_type].get(metric, 0)
            max_value = max(max_value, abs(value))
            value = averages['Serving Size Generated'][meal_type].get(metric, 0)
            max_value = max(max_value, abs(value))
            
    # Update max_value based on nutritional_metrics
    for metric in nutritional_metrics:
        for meal_type in meal_types:
            if metric in averages['Real'][meal_type]:
                max_value = max(max_value, max(abs(val) for val in averages['Real'][meal_type][metric].values()))
            if metric in averages['Generated'][meal_type]:
                max_value = max(max_value, max(abs(val) for val in averages['Generated'][meal_type][metric].values()))
            if metric in averages['Serving Size Real'][meal_type]:
                max_value = max(max_value, max(abs(val) for val in averages['Serving Size Real'][meal_type][metric].values()))
            if metric in averages['Serving Size Generated'][meal_type]:
                max_value = max(max_value, max(abs(val) for val in averages['Serving Size Generated'][meal_type][metric].values()))
    
    def exponential_transform(deviation):
        # Transform deviation to ensure positive values closer to 1 indicate smaller deviation
        a = 1  # Adjust this parameter based on the scale of your deviations
        transformed = np.exp(-a * abs(deviation))
        return transformed
    # Function to normalize values, ensuring division by max_value is safe
    def normalize(value): 
        return abs(value) / max_value if max_value else 0

    # Averages for general metrics
    avg_values_real = []
    avg_values_generated = []
    avg_values_serving_size_real = []
    avg_values_serving_size_generated = []
    for metric in general_metrics:
        if metric in ['Macronutrient Coverage', 'Micronutrient Coverage']:
            avg_real = np.mean([1-normalize(abs(averages['Real'][meal_type].get(metric, 0))) for meal_type in meal_types])
            avg_generated = np.mean([1-normalize(abs(averages['Generated'][meal_type].get(metric, 0))) for meal_type in meal_types])
            avg_serving_size_real = np.mean([1-normalize(abs(averages['Serving Size Real'][meal_type].get(metric, 0))) for meal_type in meal_types])
            avg_serving_size_generated = np.mean([1-normalize(abs(averages['Serving Size Generated'][meal_type].get(metric, 0))) for meal_type in meal_types])
            #avg_real = np.mean([inverse_transform(averages['Real'][meal_type].get(metric, 0)) for meal_type in meal_types])
            #avg_generated = np.mean([inverse_transform(averages['Generated'][meal_type].get(metric, 0)) for meal_type in meal_types])
        elif metric in ['Portion Size Deviation']:
            avg_real = np.mean([exponential_transform(averages['Real'][meal_type].get(metric, 0)) for meal_type in meal_types])
            avg_generated = np.mean([exponential_transform(averages['Generated'][meal_type].get(metric, 0)) for meal_type in meal_types])
            avg_serving_size_real = np.mean([exponential_transform(averages['Serving Size Real'][meal_type].get(metric, 0)) for meal_type in meal_types])
            avg_serving_size_generated = np.mean([exponential_transform(averages['Serving Size Generated'][meal_type].get(metric, 0)) for meal_type in meal_types])
        else:
            avg_real = np.mean([(averages['Real'][meal_type].get(metric, 0)) for meal_type in meal_types])
            avg_generated = np.mean([(averages['Generated'][meal_type].get(metric, 0)) for meal_type in meal_types])
            avg_serving_size_real = np.mean([(averages['Serving Size Real'][meal_type].get(metric, 0)) for meal_type in meal_types])
            avg_serving_size_generated = np.mean([(averages['Serving Size Generated'][meal_type].get(metric, 0)) for meal_type in meal_types])
        avg_values_real.append(avg_real)
        avg_values_generated.append(avg_generated)
        avg_values_serving_size_real.append(avg_serving_size_real)
        avg_values_serving_size_generated.append(avg_serving_size_generated)
    plot_spider_subchart(axs[0], avg_values_real, avg_values_generated, avg_values_serving_size_real, avg_values_serving_size_generated, general_metrics, "Average - General Metrics")

    # Averages for nutritional metrics
    for j, nutrient_category in enumerate(nutritional_metrics, start=1):
        avg_values_real = []
        avg_values_generated = []
        avg_values_serving_size_real = []
        avg_values_serving_size_generated = []
        nutrient_labels = set()
        for meal_type in meal_types:
            nutrient_labels.update(averages['Real'][meal_type].get(nutrient_category, {}).keys())
        nutrient_labels = sorted(list(nutrient_labels))

        for nutrient in nutrient_labels:
            avg_real = np.mean([averages['Real'][meal_type].get(nutrient_category, {}).get(nutrient, 0) for meal_type in meal_types])
            avg_generated = np.mean([averages['Generated'][meal_type].get(nutrient_category, {}).get(nutrient, 0) for meal_type in meal_types])
            avg_serving_size_real = np.mean([averages['Serving Size Real'][meal_type].get(nutrient_category, {}).get(nutrient, 0) for meal_type in meal_types])
            avg_serving_size_generated = np.mean([averages['Serving Size Generated'][meal_type].get(nutrient_category, {}).get(nutrient, 0) for meal_type in meal_types])
            avg_values_real.append(avg_real)
            avg_values_generated.append(avg_generated)
            avg_values_serving_size_real.append(avg_serving_size_real)
            avg_values_serving_size_generated.append(avg_serving_size_generated)
        plot_spider_subchart(axs[j], avg_values_real, avg_values_generated, avg_values_serving_size_real, avg_values_serving_size_generated, nutrient_labels, "Average - " + nutrient_category)

    plt.tight_layout()
    plt.savefig('tests/averages_across_meal_types_spider_plot.svg', format='svg') #dpi=1500)
    

def find_scaling_percentage(values, target_values):
    """
    Finds a scaling percentage for values so that they are as close as possible to target_values.

    :param values: Original serving size values to be scaled.
    :param target_values: The target values (real or generated) to approximate through scaling.
    :return: Scaled values and the scaling percentage rounded to the nearest 5%.
    """
    # Determine the range of target values to find a suitable scaling factor
    target_range = max(target_values) - min(target_values) if target_values else 1
    
    # Initialize variables to store the best scaling factor and its corresponding percentage
    best_scale_factor = 1
    best_percentage = 100
    
    # Iterate over possible percentages (5% steps) to find the best scaling factor
    for percentage in range(5, 101, 5):  # From 5% to 105% in steps of 5%
        scale_factor = percentage / 100.0
        scaled_values = [value * scale_factor for value in values]
        
        # Calculate how well the scaled values fit within the target range
        scaled_range = max(scaled_values) - min(scaled_values) if scaled_values else 0
        
        # Update best_scale_factor if this is the closest fit so far
        if abs(scaled_range - target_range) < abs(best_scale_factor * max(values) - target_range):
            best_scale_factor = scale_factor
            best_percentage = percentage
    
    # Apply the best scaling factor to original values
    adjusted_values = [value * best_scale_factor for value in values]
    
    return adjusted_values, best_scale_factor

# Modify the plotting function to scale the serving size values
def plot_spider_subchart(ax, values_real, values_generated, values_serving_size_real, values_serving_size_generated, labels, title):   
    # Find the best scaling percentage for real and generated separately
    #scaled_values_serving_size_real, percentage_real = find_scaling_percentage(values_serving_size_real, values_real)
    #scaled_values_serving_size_generated, percentage_generated = find_scaling_percentage(values_serving_size_generated, values_generated)

    #scaled_values_serving_size_real = values_serving_size_real
    #scaled_values_serving_size_generated = values_serving_size_generated
    num_vars = len(labels)
    angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
    angles += angles[:1]

    # Complete the loop for values
    values_real = values_real + values_real[:1]
    values_generated = values_generated + values_generated[:1]
    #scaled_values_serving_size_real = scaled_values_serving_size_real + [scaled_values_serving_size_real[0]]
    values_serving_size_generated = values_serving_size_generated + values_serving_size_generated[:1]

    ax.plot(angles, values_real, color='red', linewidth=3, linestyle='solid', label='Real')
    ax.fill(angles, values_real, color='red', alpha=0.1)
    ax.plot(angles, values_generated, color='blue', linewidth=3, linestyle='solid', label='Generated')
    ax.fill(angles, values_generated, color='blue', alpha=0.1)

    #if "General" not in title:
    #label_real = f'Serving Size Real ({percentage_real}% of Original)'
    #ax.plot(angles, scaled_values_serving_size_real, color='yellow', linewidth=3, linestyle='solid', label=label_real)
    #ax.fill(angles, scaled_values_serving_size_real, color='yellow', alpha=0.1)
    ax.plot(angles, values_serving_size_generated, color='green', linewidth=3, linestyle='solid', label='Serving Size Generated')
    ax.fill(angles, values_serving_size_generated, color='green', alpha=0.1)


    ax.set_thetagrids(np.degrees(angles[:-1]), labels, fontsize=14, color='black', rotation=45)
    ax.set_title(title, size=18, fontweight='bold', position=(0.5, 1.1))
    #ax.legend(loc='upper right', bbox_to_anchor=(1.1, 1.1))
    
    
def evaluate_meal_generation(log, meal_types):
    metric_cols = [
        'Food Coverage', #'Meal Coverage', 
        'Meal Diversity', 'Meal Realism', #'Food Overlap Similarity',
        'Macronutrient Coverage', 'Micronutrient Coverage', 
        'Macronutrient Density', 'Micronutrient Density',
        'Portion Size Deviation', 'Food Category Balance',
        'Average Category Deviations', 'Average Category Balance',
        'Average Macronutrient Coverage', 'Average Micronutrient Coverage'
    ]

    num_bootstrap = 10
    num_days_to_generate = 1000
    log.info(f"Evaluating meal generation: {num_bootstrap} bootstraps with {num_days_to_generate} meals each...")

    if not os.path.exists(f'results/evaluation/generated_meal_metrics_{num_bootstrap}_{num_days_to_generate}.pkl'): #temp
        generated_meals_metrics, serving_size_generated_meals_metrics = bootstrap_meals(metric_cols, num_bootstrap, num_days_to_generate, meal_types, log, generate_meals_func=generate_meals)
    else:
        generated_meals_metrics = pd.read_pickle(f'results/evaluation/generated_meal_metrics_{num_bootstrap}_{num_days_to_generate}.pkl')
        serving_size_generated_meals_metrics = pd.read_pickle(f'results/evaluation/generated_serving_size_meal_metrics_{num_bootstrap}_{num_days_to_generate}.pkl')

    if not os.path.exists(f'results/evaluation/real_meal_metrics_{num_bootstrap}_{num_days_to_generate}.pkl'): #temp
        real_meals_metrics = bootstrap_meals(metric_cols, num_bootstrap, num_days_to_generate, meal_types, log)
    else:
        real_meals_metrics = pd.read_pickle(f'results/evaluation/real_meal_metrics_{num_bootstrap}_{num_days_to_generate}.pkl')
        serving_size_real_meals_metrics = pd.read_pickle(f'results/evaluation/real_serving_size_meal_metrics_{num_bootstrap}_{num_days_to_generate}.pkl')

    plot_aggregated_metrics(real_meals_metrics, generated_meals_metrics, metric_cols)
    formatted_table = create_metrics_table(real_meals_metrics, generated_meals_metrics, serving_size_real_meals_metrics, serving_size_generated_meals_metrics, metric_cols, meal_types)
    print(formatted_table)
    averages = calculate_averages_for_plotting({'Real': real_meals_metrics, 'Generated': generated_meals_metrics, 'Serving Size Real': serving_size_real_meals_metrics, 'Serving Size Generated': serving_size_generated_meals_metrics}, metric_cols, meal_types)
    general_metrics = [
        'Food Coverage', 'Meal Diversity', 'Meal Realism', #'Food Overlap Similarity', 
        'Macronutrient Coverage', 'Micronutrient Coverage', 
        'Macronutrient Density', 'Micronutrient Density',
        'Portion Size Deviation', 'Food Category Balance'
    ]
    nutritional_metrics = ['Average Macronutrient Coverage', 'Average Micronutrient Coverage']
    plot_spider_chart_per_meal_type(averages, general_metrics, nutritional_metrics, meal_types)
    plot_averages_across_meal_types(averages, general_metrics, nutritional_metrics, meal_types)
    
def main():
    configure_gpu(-1)
    generate_logfile = 'logs/recommendation/recommend.log'
    generate_log = setup_logger(f'generation', generate_logfile)

    cache_dir = 'data/raw/nutrients/'
    graph_outfile = 'data/embeddings/real_and_generated_meal_embedding_graph.csv'
    embedding_outfile = 'results/embedding_model/real_and_generated_meal_embeddings.pkl'
    model_outfile = 'models/embedding_model/real_and_generated_meal_embedding_model.zip'
    meal_types = ['breakfast', 'lunch', 'dinner']
    
    
    """
    check_nutritional_data_availability(cache_dir, generate_log)
    for meal_type in meal_types:
        generate_meals(meal_type, num_days_to_generate, True, generate_log)
    create_meal_graph(graph_outfile, 'generated', generate_log)
    train_embedding_model(graph_outfile, embedding_outfile, model_outfile, generate_log)

    calorie_level = '2000 cal'
    generate_log.info(f'Getting nutrition stats for {num_days_to_generate} days')
    #aggregated_nutrition = aggregate_nutrition_per_day(meal_types, num_days_to_generate, generate_log)
    #average_similarity = compare_nutritional_values_to_targets(aggregated_nutrition, calorie_level, log)

    generate_log.info(f"Average similarity to USDA preset is {np.mean(similarities)}")
    """
    
    evaluate_meal_generation(generate_log, meal_types)

if __name__ == '__main__':
    main()