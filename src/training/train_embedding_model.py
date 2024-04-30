import os
import sys
import pandas as pd
import csrgraph as cg
from nodevectors import Node2Vec

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.helpers import *
from utils.model_utils import *
from utils.embedding_utils import *
from evaluation.evaluate_embeddings import eval_embeddings

main_categories = {
    range(1000, 2000): "Milk and Dairy",
    range(2000, 3000): "Protein Foods",
    range(3000, 4000): "Mixed Dishes",
    range(4000, 5000): "Grains",
    range(5000, 6000): "Snacks and Sweets",
    range(6000, 6400): "Fruit",
    range(6400, 7000): "Vegetables",
    range(7000, 7500): "Beverages",
    range(7500, 7600): "Alcoholic Beverages",
    range(7700, 7900): "Water",
    range(8000, 8100): "Fats and Oils",
    range(8400, 8500): "Condiments and Sauces",
    range(8800, 8900): "Sugars",
    range(9000, 9700): "Baby Foods and Formulas",
    range(9800, 10000): "Other"
}

subcategories = {
    range(1002, 1200): "Milk",
    range(1200, 1300): "Flavored Milk",
    range(1400, 1500): "Dairy Drinks and Substitutes",
    range(1600, 1700): "Cheese",
    range(1800, 1900): "Yogurt",
    range(2000, 2100): "Meats",
    range(2200, 2300): "Poultry",
    range(2400, 2500): "Seafood",
    range(2500, 2600): "Eggs",
    range(2600, 2700): "Cured Meats/Poultry",
    range(2800, 2900): "Plant-based Protein Foods",
    range(3000, 3100): "Mixed Dishes - Meat, Poultry, Seafood",
    range(3100, 3200): "Mixed Dishes - Bean/Vegetable-based",
    range(3200, 3300): "Mixed Dishes - Grain-based",
    range(3400, 3500): "Mixed Dishes - Asian",
    range(3500, 3600): "Mixed Dishes - Mexican",
    range(3600, 3700): "Mixed Dishes - Pizza",
    range(3700, 3800): "Mixed Dishes - Sandwiches",
    range(3800, 3900): "Mixed Dishes - Soups",
    range(4000, 4100): "Cooked Grains",
    range(4200, 4300): "Breads, Rolls, Tortillas",
    range(4400, 4500): "Quick Breads and Bread Products",
    range(4600, 4700): "Ready-to-Eat Cereals",
    range(4800, 4900): "Cooked Cereals",
    range(5000, 5100): "Savory Snacks",
    range(5200, 5300): "Crackers",
    range(5400, 5500): "Snack/Meal Bars",
    range(5500, 5600): "Sweet Bakery Products",
    range(5700, 5800): "Candy",
    range(5800, 5900): "Other Desserts",
    range(6000, 7000): "Fruits",
    range(6400, 6500): "Vegetables, excluding Potatoes",
    range(6800, 6900): "White Potatoes",
    range(7000, 7100): "100% Juice",
    range(7100, 7200): "Diet Beverages",
    range(7200, 7300): "Sweetened Beverages",
    range(7300, 7400): "Coffee and Tea",
    range(7500, 7600): "Alcoholic Beverages",
    range(7700, 7800): "Plain Water",
    range(7800, 7900): "Flavored or Enhanced Water",
    range(8000, 8100): "Fats and Oils",
    range(8400, 8500): "Condiments and Sauces",
    range(8800, 8900): "Sugars",
    range(9000, 9100): "Baby Foods",
    range(9200, 9300): "Baby Beverages",
    range(9400, 9500): "Infant Formulas",
    range(9600, 9700): "Human Milk",
    range(9800, 10000): "Other"
}

def standardize_columns(df):
    column_mappings = {
        'Ingredient weight (g)': 'Ingredient weight',
        'Ingredient weight': 'Ingredient weight',
        'WWEIA Category number': 'WWEIA Category code',
        'WWEIA Category code': 'WWEIA Category code'
    }
    return df.rename(columns=column_mappings)

def map_categories(code, category_map):
    for code_range, category in category_map.items():
        if code in code_range:
            return category
    return "Unknown"

def aggregate_foods(filepaths, cache_dir, log):
    if log is not None:
        log.info(f"Aggregating foods...")
    aggregated_food_outfile = 'data/mappings/food_ingredient_map.pkl'
    
    if not os.path.exists(aggregated_food_outfile):
        filepaths = sorted(filepaths, key=extract_year_from_filename, reverse=True)
        aggregated_food_data = pd.DataFrame()

        for filepath in filepaths:
            year = extract_year_from_filename(filepath)
            data = fetch_data(filepath, cache_dir, log)
            data['year'] = year 
            data = standardize_columns(data)
            if aggregated_food_data.empty:
                aggregated_food_data = data
            else:
                first_column = data.columns[0]
                data = data[~data[first_column].isin(aggregated_food_data[first_column])]
                aggregated_food_data = pd.concat([aggregated_food_data, data])     
        aggregated_food_data['WWEIA Main Category description'] = aggregated_food_data['WWEIA Category code'].apply(map_categories, args=(main_categories,))
        aggregated_food_data['WWEIA Subcategory description'] = aggregated_food_data['WWEIA Category code'].apply(map_categories, args=(subcategories,))   
        to_pickle(aggregated_food_data, aggregated_food_outfile)
    else:
        if log is not None:
            log.info(f"{aggregated_food_outfile} already exists, using cached data")
        aggregated_food_data = pd.read_pickle(aggregated_food_outfile)
        
    aggregated_food_data = aggregated_food_data[['Food code', 'Ingredient code', 'Ingredient weight']]
    aggregated_food_data['Food code'] = 'food_'+aggregated_food_data['Food code'].astype(int).astype(str)
    aggregated_food_data['Ingredient code'] = 'ingredient_'+aggregated_food_data['Ingredient code'].astype(int).astype(str)
    aggregated_food_data.columns = ['node1', 'node2', 'weight']   
    if log is not None: 
        log.info(f"Aggregated {aggregated_food_data.shape[0]} different food-to-ingredient relationships with {len(np.unique(aggregated_food_data['node1']))} unique foods and {len(np.unique(aggregated_food_data['node2']))} unique ingredients")
    return aggregated_food_data

def create_meal_graph(graph_outfile, graph_type, log):
    if log is not None:
        log.info(f"Generating meal graph...")
    ingredient_urls = [
        "https://www.ars.usda.gov/ARSUserFiles/80400530/apps/2015-2016%20FNDDS%20At%20A%20Glance%20-%20FNDDS%20Ingredients.xlsx",
        "https://www.ars.usda.gov/ARSUserFiles/80400530/apps/2017-2018%20FNDDS%20At%20A%20Glance%20-%20FNDDS%20Ingredients.xlsx",
        "https://www.ars.usda.gov/ARSUserFiles/80400530/apps/2019-2020%20FNDDS%20At%20A%20Glance%20-%20FNDDS%20Ingredients.xlsx"
    ]
    cache_dir = 'data/raw/'
    os.makedirs(cache_dir, exist_ok=True)
    
    aggregated_food_data = aggregate_foods(ingredient_urls, cache_dir+'ingredients/', log)
    aggregated_meal_data = aggregate_meals('real', log)
    if graph_type == 'generated':
        generated_meal_data =  aggregate_meals('generated', log)
        aggregated_meal_data = pd.concat([aggregated_meal_data, generated_meal_data], ignore_index=True)
    aggregated_graph_data = pd.concat([aggregated_food_data, aggregated_meal_data], ignore_index=True)
    aggregated_graph_data.to_csv(graph_outfile, index=False, header=False)

    unique_nodes = len(pd.unique(aggregated_graph_data[['node1', 'node2']].values.ravel('K')))
    
    unique_foods = set()
    unique_ingredients = set()
    for column in ['node1', 'node2']:
        print(aggregated_graph_data[column])
        unique_foods |= set(aggregated_graph_data[column][aggregated_graph_data[column].str.contains('food_', na=False)].unique())
        unique_ingredients |= set(aggregated_graph_data[column][aggregated_graph_data[column].str.contains('ingredient_', na=False)].unique())

    if log is not None:
        log.info(f"Meal-to-food-to-ingredient graph has {unique_nodes} unique nodes and {aggregated_graph_data.shape[0]} edges")
        log.info(f"{unique_foods} unique foods")
        log.info(f"{unique_ingredients} unique ingredients")
            
def train_embedding_model(graph_file, embedding_outfile, model_outfile, log):
    if log is not None:
        log.info(f"Training embedding model...")
    G = cg.read_edgelist(graph_file, directed=False, sep=',')
    config = load_config('config.json')
    
    g2v = Node2Vec(
        n_components=config['embedding_model']['embedding_size'],
        walklen=config['embedding_model']['walk_length'],
        keep_walks=True
    )
    g2v.fit(G)
    if model_outfile is not None:
        g2v.save(model_outfile.split('.')[0])
    embedding_dict = {node: g2v.predict(node) for node in G.nodes()}
    embedding_df = convert_dict_to_embedding_df(embedding_dict)
    to_pickle(embedding_df, embedding_outfile)

    if log is not None:
        log.info(f"Embedding model training finished")
        log.info(f"Graph embeddings saved to {embedding_outfile}")
        log.info(f"Embedding model  saved to {model_outfile}")

    return embedding_df
    
def main():    
    embedding_model_log = setup_logger('embedding_model_log', f'logs/embedding_model/training.log')
    graph_outfile = 'data/embeddings/real_meal_embedding_graph.csv'
    embedding_outfile = 'results/embedding_model/real_meal_embeddings.pkl'
    model_outfile = f'models/embedding_model/real_meal_embedding_model.zip'
    
    create_meal_graph(graph_outfile, 'real', embedding_model_log)
    embedding_df = train_embedding_model(graph_outfile, embedding_outfile, model_outfile, embedding_model_log)
    eval_embeddings(graph_outfile, embedding_df, 128, "Node2Vec", embedding_model_log)
    create_gephi_graph()

if __name__ == '__main__':
    main()

