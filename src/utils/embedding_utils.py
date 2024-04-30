import os
import sys
import random
import numpy as np
import pandas as pd
import networkx as nx

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.helpers import *

def convert_dict_to_embedding_df(embedding_dict):
    embedding_df = pd.DataFrame.from_dict(embedding_dict, orient='index')
    embedding_df.reset_index(inplace=True)
    embedding_df.columns = ['node'] + [f'emb_{i}' for i in range(embedding_df.shape[1] - 1)]
    return embedding_df

def load_food_mapping(filepath):
    mapping_df = pd.read_pickle(filepath)
    mapping_dict = dict(zip(mapping_df['Food code'], mapping_df['Main food description']))
    return mapping_dict

def format_meals(meal_df, meal_type, meal_origin, log):
    if log is not None:
        log.info(f"Formatting {meal_origin} {meal_type} meals...")
    formatted_meal_data = pd.DataFrame()
    file_identifier = meal_type.split('_')[0]
    for index, row in meal_df.iterrows():
        if meal_origin == 'real':
            meal_name = f'{meal_origin}_{file_identifier}_{int(index)+1}'
        else:
            meal_name = f'{meal_origin}_{index}'
        for food_code, gram_amount in row.items():
            if gram_amount > 0:
                meal_data = {'meal': meal_name, 'food_code': f'food_{int(food_code)}', 'gram_amount': gram_amount}
                formatted_meal_data = pd.concat([formatted_meal_data, pd.DataFrame([meal_data])], ignore_index=True)
                
    cache_dir = f'data/processed/{meal_origin}_meals/graphs'
    os.makedirs(cache_dir, exist_ok=True)
    to_pickle(formatted_meal_data, f'{cache_dir}/{meal_type}_meal_graph.pkl')
    return formatted_meal_data
    
def aggregate_meals(meal_origin, log):
    if log is not None:
        log.info(f"Aggregating {meal_origin} meals...")
    meal_to_food_map_outfile = f'data/mappings/graph_meal_labels_to_food_{meal_origin}.pkl'
    aggregated_meal_outfile = f'data/embeddings/aggregated_meal_data_{meal_origin}.pkl'
    
    directory = f'data/processed/{meal_origin}_meals/dataframes'
    if meal_origin == 'generated':
        files = [f for f in os.listdir(directory) if f.endswith('.pkl')]
    else:
        files = [f for f in os.listdir(directory) if 'sparsity_filtered' in f and f.endswith('.pkl')]
    aggregated_meal_data = pd.DataFrame()
    for file in files:
        file_path = os.path.join(directory, file)
        df = pd.read_pickle(file_path)
        meal_type = os.path.splitext(os.path.basename(file))[0]
        formatted_meal_data = format_meals(df, meal_type, meal_origin, log)
        aggregated_meal_data = pd.concat([aggregated_meal_data, formatted_meal_data], ignore_index=True)

    meal_to_food_map = aggregated_meal_data.groupby('meal')['food_code'].apply(list).to_dict()
    to_pickle(meal_to_food_map, meal_to_food_map_outfile)
    to_pickle(aggregated_meal_data, aggregated_meal_outfile)

    aggregated_meal_data.columns = ['node1', 'node2', 'weight']
    if log is not None:
        log.info(f"Aggregated {meal_origin} meals")
        log.info(f"Saving meal label-to-food map in {meal_to_food_map_outfile}")
        log.info(f"Saving aggregated food code data in {aggregated_meal_outfile}")
        log.info(f"Aggregated {aggregated_meal_data.shape[0]} different meal-to-food relationships with {len(np.unique(aggregated_meal_data['node1']))} unique meals and {len(np.unique(aggregated_meal_data['node2']))} unique foods")

    return aggregated_meal_data

def create_gephi_graph():
    food_ingredient_df = pd.read_pickle('data/mappings/food_ingredient_map.pkl')
    aggregated_meal_data_df = pd.read_pickle('data/embeddings/aggregated_meal_data_real.pkl')

    food_descriptions = {code: desc.split(',')[0] for code, desc in zip(food_ingredient_df['Food code'], food_ingredient_df['Main food description'])}
    ingredient_descriptions = {code: desc.split(',')[0] for code, desc in zip(food_ingredient_df['Ingredient code'], food_ingredient_df['Ingredient description'])}

    G = nx.Graph()
    for _, row in food_ingredient_df.iterrows():
        food_code, ingredient_code, weight = row['Food code'], row['Ingredient code'], row['Ingredient weight']
        food_desc = food_descriptions.get(food_code, 'Unknown Food')
        ingredient_desc = ingredient_descriptions.get(ingredient_code, 'Unknown Ingredient')
        G.add_node(food_desc, category='food')
        G.add_node(ingredient_desc, category='ingredient')
        G.add_edge(food_desc, ingredient_desc, weight=weight)

    aggregated_meal_data_df.columns = ['node1', 'node2', 'weight']
    for _, row in aggregated_meal_data_df.iterrows():
        meal_label, food_code, weight = row['node1'], row['node2'], row['weight']
        meal_category = meal_label.split('_')[1]
        meal_label = meal_label.split('_')[1].capitalize() + ' Meal ' + meal_label.split('_')[-1]
        food_desc = food_descriptions.get(food_code.split('_')[-1], 'Unknown Food')
        G.add_node(food_desc, category='food')  
        G.add_node(meal_label, category=meal_category)
        G.add_edge(meal_label, food_desc, weight=weight)

    sampling_percentages = {'breakfast': 3, 'lunch': 3, 'dinner': 3, 'food': 40, 'ingredient': 100}

    sampled_nodes = []
    for category, percentage in sampling_percentages.items():
        category_nodes = [node for node, attr in G.nodes(data=True) if attr['category'] == category]
        sample_size = int(len(category_nodes) * (percentage / 100))
        sampled_nodes.extend(random.sample(category_nodes, min(sample_size, len(category_nodes))))

    subG = nx.Graph(G.subgraph(sampled_nodes))

    isolated_meals = [node for node in subG.nodes if subG.degree(node) == 0]
    subG.remove_nodes_from(isolated_meals)

    nx.write_gexf(subG, 'results/embedding_model/gephi_graph_with_descriptions.gexf')