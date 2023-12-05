import os
import sys
import numpy as np
import pandas as pd

import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Input, Dense, Lambda, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam

from sklearn.metrics import mean_squared_error, r2_score
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, average_precision_score

import wandb

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.helpers import *

class PresenceTrainingLogger(tf.keras.callbacks.Callback):
    def __init__(self, model, X_val, log):
        super(PresenceTrainingLogger, self).__init__()
        self.model = model
        self.X_val = X_val
        self.log = log
    def on_epoch_end(self, epoch, logs=None):
        if logs is not None:
            self.log.info(f"Epoch {epoch + 1:<3}: Training Loss: {logs.get('loss'):14.10f}, "
                         f"Validation Loss: {logs.get('val_loss'):14.10f}, "
                         f"Training KL Loss: {logs.get('kl_loss'):14.10f}, "
                         f"Validation KL Loss: {logs.get('val_kl_loss'):14.10f}")
            wandb.log({"epoch": epoch, **logs})

class PortionTrainingLogger(tf.keras.callbacks.Callback):      
    def __init__(self, log):
        super(PortionTrainingLogger, self).__init__()
        self.log = log 
    def on_epoch_end(self, epoch, logs=None):
        if logs is not None:
            self.log.info(f"Epoch {epoch + 1:<3}: Training Loss: {logs.get('loss'):>8.10f}, "
                         f"Validation Loss: {logs.get('val_loss'):>8.10f}, "
                         f"Training R-squared: {logs.get('r_squared'):>8.10f}, "
                         f"Validation R-squared: {logs.get('val_r_squared'):>8.10f}")
            wandb.log({"epoch": epoch, **logs})
            
class KLAnnealingCallback(tf.keras.callbacks.Callback):
    def __init__(self, initial_epochs=15, increment=0.01):
        self.initial_epochs = initial_epochs
        self.increment = increment

    def on_epoch_end(self, epoch, logs=None):
        if epoch < self.initial_epochs:
            K.set_value(PresenceModel.kl_weight, 0.0)
        else:
            new_value = K.get_value(PresenceModel.kl_weight) + self.increment
            K.set_value(PresenceModel.kl_weight, new_value)
            
class VAELossLayer(tf.keras.layers.Layer):
    def __init__(self, config, **kwargs):
        super(VAELossLayer, self).__init__(**kwargs)
        #self.alpha = config["presence_model"]["alpha"]
    
    def call(self, inputs):
        x, z_mean, z_log_var, y_true = inputs
        recon_loss = K.sum(K.binary_crossentropy(y_true, x), axis=-1)
        kl_loss = -0.5 * K.sum(1 + z_log_var - K.square(z_mean) - K.exp(z_log_var), axis=-1)
        #similarity_loss = K.mean((K.dot(K.l2_normalize(z_mean, axis=1), K.transpose(K.l2_normalize(z_mean, axis=1))) - 1) / 2)
        self.add_metric(recon_loss, name='recon_loss', aggregation='mean')
        self.add_metric(kl_loss, name='kl_loss', aggregation='mean')
        #self.add_metric(similarity_loss, name='sim_loss', aggregation='mean')
        total_loss = recon_loss + (PresenceModel.kl_weight * kl_loss) #+ (self.alpha * similarity_loss)
        self.add_loss(total_loss)
        return x

class PortionModel:
    def __init__(self, input_size, meal_type, config_path):
        self.config = load_config(config_path)
        self.model = self.build_model(input_size)
        self.model_save_path = construct_save_path(meal_type, 'portion_model', 'model', self.config) 
        self.results_save_path = construct_save_path(meal_type, 'portion_model', 'results', self.config) 
    
    def build_model(self, input_shape):
        def rmse_loss(y_true, y_pred):
            return K.sqrt(K.mean(K.square(y_pred - y_true)))

        def r_squared(y_true, y_pred):
            ss_res = K.sum(K.square(y_true - y_pred))
            ss_tot = K.sum(K.square(y_true - K.mean(y_true)))
            return (1 - ss_res/(ss_tot + K.epsilon()))

        model = Sequential()
        model.add(Dense(self.config["portion_model"]["layer_sizes"][0], activation='sigmoid', input_shape=(input_shape,)))
        for size in self.config["portion_model"]["layer_sizes"][1:]:
            model.add(Dropout(self.config["portion_model"]["dropout_rate"]))
            model.add(Dense(size, activation='relu'))
        model.add(Dense(input_shape))
        model.compile(optimizer='adam', loss=rmse_loss, metrics=[r_squared])
        return model

    def train(self, X_train, y_train, X_val, y_val, log):
        early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
        training_logger = PortionTrainingLogger(log)
        self.model.fit(
            X_train, 
            y_train, 
            epochs=self.config["portion_model"]["epochs"], 
            batch_size=self.config["portion_model"]["batch_size"], 
            validation_data=(X_val, y_val), 
            callbacks=[early_stopping, training_logger],
            verbose=0
        )

    def evaluate(self, X_test, y_test):
        y_pred = self.model.predict(X_test)
        y_pred_df = pd.DataFrame(y_pred, index=y_test.index, columns=y_test.columns)

        rmses, r2s, pearson_corrs = [], [], []
        for i in range(y_test.shape[0]):
            row_y_test = y_test.iloc[i, :]
            row_y_pred = y_pred_df.iloc[i, :]

            rmses.append(np.sqrt(mean_squared_error(row_y_test, row_y_pred)))
            r2s.append(r2_score(row_y_test, row_y_pred))
            
            if np.std(row_y_test) > 0 and np.std(row_y_pred) > 0:
                pearson_corrs.append(np.corrcoef(row_y_test, row_y_pred)[0, 1])
            else:
                pearson_corrs.append(np.nan)  # Assign NaN if std is zero
                
        avg_rmse = np.nanmean(rmses)
        avg_r2 = np.nanmean(r2s)
        avg_pearson_corr = np.nanmean(pearson_corrs)
        agg_rmse = np.sqrt(mean_squared_error(y_test.values.flatten(), y_pred_df.values.flatten()))
        agg_r2 = r2_score(y_test.values.flatten(), y_pred_df.values.flatten())
        agg_pearson_corr = np.corrcoef(y_test.values.flatten(), y_pred_df.values.flatten())[0, 1]
        micro_results = {"r2": agg_r2, "rmse": agg_rmse, "pcc": agg_pearson_corr}
        macro_results = {"r2": avg_r2, "rmse": avg_rmse, "pcc": avg_pearson_corr}

        return micro_results, macro_results

    def save_model(self, log):
        self.model.save(self.model_save_path + 'model.keras')
        log.info(f"Model saved to {self.model_save_path}model.keras")

class PresenceModel:
    def __init__(self, input_size, meal_type, config_path):
        self.config = load_config(config_path)
        self.input_size = input_size
        PresenceModel.kl_weight = K.variable(0.0)
        self.model = self.build_model()
        self.model_save_path = construct_save_path(meal_type, 'presence_model', 'model', self.config) 
        self.results_save_path = construct_save_path(meal_type, 'presence_model', 'results', self.config) 
        
    def sampling(self, args):
        z_mean, z_log_var = args
        batch = K.shape(z_mean)[0]
        dim = K.int_shape(z_mean)[1]
        epsilon = K.random_normal(shape=(batch, dim), mean=0., stddev=0.5)
        return z_mean + K.exp(0.5 * z_log_var) * epsilon

    def build_model(self):
        encoder_inputs = Input(shape=(self.input_size,))
        encoder = encoder_inputs
        for size in self.config["presence_model"]["encoder_sizes"]:
            encoder = Dense(size, activation='relu')(encoder)
            encoder = Dropout(self.config["presence_model"]["dropout_rate"])(encoder)
            
        z_mean = Dense(self.config["presence_model"]["latent_dim"], name='z_mean')(encoder)
        z_log_var = Dense(self.config["presence_model"]["latent_dim"], name='z_log_var')(encoder)
        z = Lambda(self.sampling, output_shape=(self.config["presence_model"]["latent_dim"],), name='z')([z_mean, z_log_var])
        decoder = z
        for size in self.config["presence_model"]["decoder_sizes"]:
            decoder = Dense(size, activation='relu')(decoder)
        ingredient_selection = Dense(self.input_size, activation='sigmoid', name='food_selection_output')(decoder)

        vae_output = VAELossLayer(self.config)([ingredient_selection, z_mean, z_log_var, encoder_inputs])
        vae = Model(inputs=encoder_inputs, outputs=vae_output)
        vae.compile(optimizer=Adam(self.config["presence_model"]["learning_rate"]))
        return vae
    
    def train(self, X_train, X_train_temp, X_val, X_val_temp, log):
        kl_annealing = KLAnnealingCallback()
        early_stopping = EarlyStopping(monitor='val_loss', patience=100, restore_best_weights=False)
        training_logger = PresenceTrainingLogger(self.model, X_val, log)
        self.model.fit(
            X_train, X_train,
            batch_size=self.config["presence_model"]["batch_size"],
            epochs=self.config["presence_model"]["epochs"],
            validation_data=(X_val, X_val),
            callbacks=[kl_annealing, early_stopping, training_logger],
            verbose=0
        )
    
    def evaluate(self, X_test, X_test_temp):
        y_pred = self.model.predict(X_test)
        y_pred_binary = np.where(y_pred > 0.5, 1, 0)

        precision_macro = precision_score(X_test, y_pred_binary, average='macro')
        recall_macro = recall_score(X_test, y_pred_binary, average='macro')
        f1_macro = f1_score(X_test, y_pred_binary, average='macro')

        precision_micro = precision_score(X_test, y_pred_binary, average='micro')
        recall_micro = recall_score(X_test, y_pred_binary, average='micro')
        f1_micro = f1_score(X_test, y_pred_binary, average='micro')

        #roc_auc_macro = roc_auc_score(X_test, y_pred, average='macro', multi_class='ovr')
        roc_auc_micro = roc_auc_score(X_test, y_pred, average='micro', multi_class='ovo')
        #auprc_macro = average_precision_score(X_test, y_pred, average='macro')
        auprc_micro = average_precision_score(X_test, y_pred, average='micro')

        micro_results = {
            'precision': precision_micro, 'recall': recall_micro, 'f1': f1_micro,
            'roc_auc': roc_auc_micro, 'auprc': auprc_micro
        }
        macro_results = {
            'precision': precision_macro, 'recall': recall_macro, 'f1': f1_macro, 
            'roc_auc': 0, 'auprc': 0
        } # can't get macro aucs here due to sparse data
        
        return micro_results, macro_results
    
    def save_model(self, log):
        self.model.save(self.model_save_path + 'model.keras')
        log.info(f"Model saved to {self.model_save_path}model.keras")
        
def construct_save_path(meal_type, model_type, save_type, config):
    if model_type == 'portion_model':
        logdir = '_'.join(str(size) for size in config[model_type]["layer_sizes"])
    elif model_type == 'presence_model':
        logdir = f'{config[model_type]["learning_rate"]}_{config[model_type]["latent_dim"]}'
    logdir += f'_{config[model_type]["epochs"]}_{config[model_type]["batch_size"]}'
    
    if save_type == 'model':
        path = f'models/{model_type}/{meal_type}/{logdir}/'
    elif save_type == 'results':
        path = f'results/{model_type}/{meal_type}/{logdir}/'
    elif save_type == 'logging':
        path = f'logs/{model_type}/{meal_type}/{logdir}/'
    os.makedirs(path, exist_ok=True)
    
    return path