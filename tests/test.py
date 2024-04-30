import os
import sys
import pandas as pd

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.utils.helpers import *
from src.utils.models import *
from src.training.train_portion_model import *

def main():
    configure_gpu(-1)

    meal_type = sys.argv[1]
    data = pd.read_pickle(f'data/processed/meals/dataframes/{meal_type}_sparsity_filtered.pkl')
    fold_data = pd.read_pickle(f'results/presence_model/{meal_type}/0.001_20_200_64/fold_5_indices.pkl')
    #columns = pd.read_pickle(f'data/processed/meals/dataframes/sparsity_filtration/{meal_type}/{meal_type}_sparse_85.pkl').columns    
    print(fold_data)

    train_indices = fold_data['train_index']
    val_indices = fold_data['val_index']
    y_train = data.iloc[train_indices]
    y_val = data.iloc[val_indices]
    
    #y_train = y_train[columns]
    #y_val = y_val[columns]
    
    X_train = (y_train > 0).astype(int)
    X_val = (y_val > 0).astype(int)

    print(X_train.shape)
    model = PresenceModel(X_train.shape[1], meal_type, 'config.json')
    
    model.train(X_train, y_train, X_val, y_val, None)
    model.model_save_path = './'
    model.save_models()
    print(model.model.summary())
    
    
    # Load Models using the static method
    loaded_vae = PresenceModel.load_model(model.model_save_path + 'vae_model.keras')
    loaded_decoder = PresenceModel.load_model(model.model_save_path + 'decoder_model.keras')

    print(loaded_vae.summary())
    print(loaded_decoder.summary())

main()