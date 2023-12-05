import os
import re
import sys
import pandas as pd

import csrgraph as cg
from nodevectors import Node2Vec

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.helpers import *
from utils.model_utils import *
from evaluation.evaluate_embeddings import eval_embeddings

def convert_dict_to_embedding_df(embedding_dict):
    embedding_df = pd.DataFrame.from_dict(embedding_dict, orient='index')
    embedding_df.reset_index(inplace=True)
    embedding_df.columns = ['node'] + [f'emb_{i}' for i in range(embedding_df.shape[1] - 1)]
    return embedding_df

def extract_year_from_filename(filename):
    match = re.search(r'(\d{4})-', filename)
    return int(match.group(1)) if match else None

def standardize_columns(df):
    column_mappings = {
        'Ingredient weight (g)': 'Ingredient weight',
        'Ingredient weight': 'Ingredient weight',
        'WWEIA Category number': 'WWEIA Category code',
        'WWEIA Category code': 'WWEIA Category code'
    }
    return df.rename(columns=column_mappings)

def aggregate_meals(embedding_model_log):
    embedding_model_log.info("Aggregating meals...")
    meal_to_food_map_outfile = 'data/mappings/graph_meal_labels_to_food.pkl'
    aggregated_meal_outfile = 'data/embeddings/aggregated_meal_data.pkl'
    
    if not os.path.exists(meal_to_food_map_outfile) or not os.path.exists(aggregated_meal_outfile):
        directory = 'data/processed/meals/dataframes/'
        files = [f for f in os.listdir(directory) if 'sparsity_filtered' in f and f.endswith('.pkl')]
        aggregated_meal_data = pd.DataFrame()
        
        for file in files:
            file_path = os.path.join(directory, file)
            df = pd.read_pickle(file_path)
            file_identifier = os.path.splitext(os.path.basename(file))[0].split('_')[0]
            df.index = df.index.astype(str) + '_' + file_identifier
            df_long = df.reset_index().melt(id_vars='index', var_name='food_code', value_name='gram_amount')
            df_long.rename(columns={'index': 'meal'}, inplace=True)
            df_long = df_long[df_long['gram_amount'] > 0]
            aggregated_meal_data = pd.concat([aggregated_meal_data, df_long], ignore_index=True)
        
        aggregated_meal_data['food_code'] = aggregated_meal_data['food_code'].astype(int)
        aggregated_meal_data.columns = ['node1', 'node2', 'weight']
        
        meal_to_food_map = aggregated_meal_data.groupby('node1')['node2'].apply(list).to_dict()
        embedding_model_log.info(f"Saving meal label-to-food map in {meal_to_food_map_outfile}")
        embedding_model_log.info(f"Saving aggregated food code data in {aggregated_meal_outfile}")
        to_pickle(meal_to_food_map, meal_to_food_map_outfile)
        to_pickle(aggregated_meal_data, aggregated_meal_outfile)
    else:
        embedding_model_log.info(f"{aggregated_meal_outfile} already exists, using cached data")
        meal_to_food_map = pd.read_pickle(meal_to_food_map_outfile)
        aggregated_meal_data = pd.read_pickle(aggregated_meal_outfile)
        
    embedding_model_log.info(f"Aggregated {aggregated_meal_data.shape[0]} different meal-to-food relationships with {len(meal_to_food_map)} unique meals")

    return aggregated_meal_data

def aggregate_foods(filepaths, cache_dir, embedding_model_log):
    embedding_model_log.info(f"Aggregating foods...")
    aggregated_food_outfile = 'data/mappings/food_ingredient_map.pkl'
    
    if not os.path.exists(aggregated_food_outfile):
        filepaths = sorted(filepaths, key=extract_year_from_filename, reverse=True)
        aggregated_food_data = pd.DataFrame()

        for filepath in filepaths:
            year = extract_year_from_filename(filepath)
            data = fetch_data(filepath, cache_dir, embedding_model_log)
            data['year'] = year 
            data = standardize_columns(data)
            if aggregated_food_data.empty:
                aggregated_food_data = data
            else:
                first_column = data.columns[0]
                data = data[~data[first_column].isin(aggregated_food_data[first_column])]
                aggregated_food_data = pd.concat([aggregated_food_data, data])
        
        aggregated_food_data = aggregated_food_data[['Food code', 'Ingredient code', 'Ingredient weight']]
        aggregated_food_data.columns = ['node1', 'node2', 'weight']
        to_pickle(aggregated_food_data, aggregated_food_outfile)
    else:
        embedding_model_log.info(f"{aggregated_food_outfile} already exists, using cached data")
        aggregated_food_data = pd.read_pickle(aggregated_food_outfile)
        
    embedding_model_log.info(f"Aggregated {aggregated_food_data.shape[0]} different food-to-ingredient relationships with {len(np.unique(aggregated_food_data['node1']))} unique foods and {len(np.unique(aggregated_food_data['node2']))} unique ingredients")

    return aggregated_food_data

def create_meal_graph(filepath, embedding_model_log):
    embedding_model_log.info(f"Generating meal graph...")
    if not os.path.exists(filepath):
        ingredient_urls = [
            "https://www.ars.usda.gov/ARSUserFiles/80400530/apps/2015-2016%20FNDDS%20At%20A%20Glance%20-%20FNDDS%20Ingredients.xlsx",
            "https://www.ars.usda.gov/ARSUserFiles/80400530/apps/2017-2018%20FNDDS%20At%20A%20Glance%20-%20FNDDS%20Ingredients.xlsx",
            "https://www.ars.usda.gov/ARSUserFiles/80400530/apps/2019-2020%20FNDDS%20At%20A%20Glance%20-%20FNDDS%20Ingredients.xlsx"
        ]
        cache_dir = 'data/raw/'
        os.makedirs(cache_dir, exist_ok=True)
        
        aggregated_food_data = aggregate_foods(ingredient_urls, cache_dir+'ingredients/', embedding_model_log)
        aggregated_meal_data = aggregate_meals(embedding_model_log)
        aggregated_graph_data = pd.concat([aggregated_food_data, aggregated_meal_data], ignore_index=True)
        aggregated_graph_data.to_csv(filepath, index=False, header=False)
    else:
        embedding_model_log.info(f"{filepath} already exists, using cached data")
        aggregated_graph_data = pd.read_csv(filepath, index_col=False, header=None)
        aggregated_graph_data.columns = ['node1', 'node2', 'weight']
        
    unique_nodes = len(pd.unique(aggregated_graph_data[['node1', 'node2']].values.ravel('K')))
    embedding_model_log.info(f"Meal-to-food-to-ingredient graph has {unique_nodes} unique nodes and {aggregated_graph_data.shape[0]} edges")

def train_embedding_model(filepath, embedding_model_log):
    embedding_model_log.info(f"Training embedding model...")
    G = cg.read_edgelist(filepath, directed=False, sep=',')
    config = load_config('config.json')
    
    savedir = 'models/embedding_model/'
    model_outfile = f'{savedir}node2vec_embedding_model.zip'
    os.makedirs(savedir, exist_ok=True)
    if not os.path.exists(model_outfile):
        g2v = Node2Vec(
            n_components=config['embedding_model']['embedding_size'],
            walklen=config['embedding_model']['walk_length'],
            keep_walks=True
        )
        g2v.fit(G)
        g2v.save(model_outfile.split('.')[0])
    else:
        g2v = Node2Vec.load(model_outfile)
        
    embedding_model_log.info(f"Embedding model training finished")
    embeddings = {node: g2v.predict(node) for node in G.nodes()}
    embedding_outfile = 'models/embedding_model/graph_embeddings.pkl'
    embedding_model_log.info(f"Graph embeddings saved to {embedding_outfile}")
    to_pickle(embeddings, embedding_outfile)
    
    return embeddings
    
def main():    
    embedding_model_log = setup_logger('embedding_model_log', f'logs/embedding_model_training.log')
    filepath = 'data/embeddings/embedding_graph.csv'
    create_meal_graph(filepath, embedding_model_log)
    embeddings_dict = train_embedding_model(filepath, embedding_model_log)
    embedding_df = convert_dict_to_embedding_df(embeddings_dict)
    eval_embeddings(filepath, embedding_df, 128, "Node2Vec", embedding_model_log)

if __name__ == '__main__':
    main()

