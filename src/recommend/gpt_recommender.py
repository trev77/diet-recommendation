import os
import re
import sys
import json
import pandas as pd

from openai import OpenAI
from dotenv import load_dotenv
load_dotenv()
#print(os.environ['OPENAI_API_KEY'])
client = OpenAI()

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from evaluation.evaluate_generation import calculate_meal_nutrition_from_df

#text_model = 'gpt-4'

def initialize_config(json_file_path, variable_name):
    with open(json_file_path, 'r') as json_file:
        json_data = json.load(json_file)
    globals()[variable_name] = json_data
    
initialize_config('config.json', 'config')

class Meal:
    def __init__(self, respondent, intake_dow, food_source, home_eaten, foods, grams,):
        self.seqn = respondent
        self.day = intake_dow
        self.fdsc = food_source
        self.home = home_eaten
        self.fdcd = foods
        self.grms = grams

"""
def get_gpt_embeddings(text_to_embed, embedding_model="text-embedding-ada-002"):
    text_to_embed = text_to_embed.replace("\n", " ")
    response = openai.Embedding.create(
        input=[text_to_embed],
        model=embedding_model
    )
    embeddings = response['data'][0]['embedding']
    #print(embeddings)
    #print(len(embeddings))
    return embeddings
"""

def get_gpt_recommendation(meal_type,  text_model='gpt-4'):
    meal_data = pd.read_pickle(f'data/processed/real_meals/dataframes/{meal_type}_sparsity_filtered.pkl')
    food_to_nutrient = pd.read_pickle('data/mappings/food_to_nutrient.pkl')

    filtered_food_to_nutrient = food_to_nutrient[food_to_nutrient['Food code'].isin(list(meal_data.columns))]
    mapping_dict = pd.Series(filtered_food_to_nutrient['Main food description'].values,
                         index=filtered_food_to_nutrient['Food code']).to_dict()
    
    messages = [{
        "role": "system", 
        "content": "You are a diet recommendation system."
    }]
    message = f'Recommend me a {meal_type} meal in the format <food code>:<food description>:<gram amount>g` using the following dictionary of foods: {mapping_dict} with format <food code>: <food description>. Only express the measurements in grams. No other text is necessary.'
    messages.append(
        {"role": "user", "content": message},
    )
    chat = client.chat.completions.create(
        model=text_model, 
        messages=messages,
    )
    reply = chat.choices[0].message.content
    display_reply = reply.replace("\n", "\n\t")

    print("<{}>:\n\t{}:".format(text_model, display_reply))
    messages.append({"role": "diet recommendation system", "content": reply})

    return reply

def generate_meals_dataframe(meal_combinations):
    meals_list = []

    # Assume each meal is separated by a newline character
    for meal_combination in meal_combinations.strip().split('\n'):
        meal_dict = {}
        items = meal_combination.split(':')
        food_code, portion_with_g = items[0], items[2]
        portion = float(re.sub(r'[^\d.]+', '', portion_with_g))
        meal_dict[food_code] = portion
        if meal_dict:
            meals_list.append(meal_dict)
    meal_df = pd.DataFrame(meals_list).fillna(0)
    
    return meal_df

def main():
    meal_type = 'breakfast'
    num_combos = 1
    meal_nutrition = []
    for _ in range(num_combos):
        meal = get_gpt_recommendation(meal_type)
        meal_df = generate_meals_dataframe(meal)
        meal_df = meal_df.sum(axis=0)
        nutrient_data = pd.read_pickle('data/mappings/food_to_nutrient.pkl')
        nutrient_data.set_index('Food code', inplace=True)
        nutrient_data = nutrient_data.apply(pd.to_numeric, errors='coerce').fillna(0)
        meal_nutrients = calculate_meal_nutrition_from_df(meal_df, nutrient_data)
        meal_nutrition.append(meal_nutrients)
    nutrition_df = pd.DataFrame(meal_nutrition)
    print(nutrition_df)
    """
    meal_embeddings = pd.read_csv('embeddings/results/{}_meal_embeddings_tsne_{}_{}_{}.csv'.format(
        meal_type,
        config['embeddings'][meal_type]['algo'],
        config['embeddings'][meal_type]['normalized'],
        config['embeddings'][meal_type]['embedding_size'])
    )
    print(meal_embeddings)
    descrips = meal_embeddings.iloc[np.where(meal_embeddings['labels'] == 'real')]['descrip']

    descrips['descrip'] = meal_embeddings['descrip'].apply(ast.literal_eval)

    meals_clean = []
    for meal in descrips['descrip']:
        meal_str = ''
        for food in meal:
            meal_str += food + ' '
        meals_clean.append(meal_str)
    print(meals_clean)

    for text_to_embed in meals_clean:
        get_gpt_embeddings(text_to_embed, embedding_model='text-embedding-ada-002')
    """
    
main()