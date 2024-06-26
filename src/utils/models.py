import os
import sys
import numpy as np
import pandas as pd

import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras.models import Sequential, Model, load_model
from tensorflow.keras.layers import Input, Dense, Lambda, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam

from sklearn.metrics import mean_squared_error, r2_score
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, average_precision_score

import wandb

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.helpers import *

class PresenceTrainingLogger(tf.keras.callbacks.Callback):
    def __init__(self, model, X_val, log=None):  
        super(PresenceTrainingLogger, self).__init__()
        self.model = model
        self.X_val = X_val
        self.log = log

    def on_epoch_end(self, epoch, logs=None):
        if self.log is not None and logs is not None:
            self.log.info(f"Epoch {epoch + 1:<3}: Training Loss: {logs.get('loss'):14.10f}, "
                          f"Validation Loss: {logs.get('val_loss'):14.10f}, "
                          f"Training KL Loss: {logs.get('kl_loss'):14.10f}, "
                          f"Validation KL Loss: {logs.get('val_kl_loss'):14.10f}")
            wandb.log({"epoch": epoch, **logs})
class PortionTrainingLogger(tf.keras.callbacks.Callback):
    def __init__(self, log=None): 
        super(PortionTrainingLogger, self).__init__()
        self.log = log

    def on_epoch_end(self, epoch, logs=None):
        if self.log is not None and logs is not None:
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
        
"""
class VAELossLayer(tf.keras.layers.Layer):
    def __init__(self, config, **kwargs):
        super(VAELossLayer, self).__init__(**kwargs)
        self.config = config

    def call(self, inputs):
        x, z_mean, z_log_var, y_true = inputs
        # Reconstruction loss
        reconstruction_loss = K.sum(K.binary_crossentropy(y_true, x), axis=-1)
        
        # KL divergence loss
        kl_loss = -0.5 * K.sum(1 + z_log_var - K.square(z_mean) - K.exp(z_log_var), axis=-1)
        
        # Custom Metrics
        jaccard_loss = jaccard_index_loss(y_true, x)
        gini_loss = gini_coefficient(y_true, x)
        coverage_loss = food_coverage_loss(x)

        # Configurable weights for each loss component
        total_loss = reconstruction_loss + (PresenceModel.kl_weight * kl_loss)
        
        total_loss += self.config['presence_model']['alpha'] * jaccard_loss
        total_loss += self.config['presence_model']['beta'] * gini_loss
        total_loss += self.config['presence_model']['gamma'] * coverage_loss
        self.add_loss(total_loss, inputs=inputs)

        self.add_metric(reconstruction_loss, name='recon_loss', aggregation='mean')
        self.add_metric(kl_loss, name='kl_loss', aggregation='mean')
        self.add_metric(jaccard_loss, name='jaccard_loss', aggregation='mean')
        self.add_metric(gini_loss, name='gini_loss', aggregation='mean')
        self.add_metric(coverage_loss, name='coverage_loss', aggregation='mean')

        return x
""" 

class PortionModel:
    def __init__(self, input_size, meal_type, config_path):
        self.config = load_config(config_path)
        self.model = self.build_model(input_size)
        self.model_save_path = construct_save_path(meal_type, 'portion_model', 'model', self.config) 
        self.results_save_path = construct_save_path(meal_type, 'portion_model', 'results', self.config) 
    
    def build_model(self, input_shape):
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

    def save_model(self, log=None):
        self.model.save(self.model_save_path + 'model.keras')
        if log is not None:
            log.info(f"Model saved to {self.model_save_path}model.keras")

    @staticmethod
    def load_model(filepath):
        custom_objects = {'rmse_loss': rmse_loss, 'r_squared': r_squared}
        return load_model(filepath, custom_objects=custom_objects)

class PresenceModel:
    def __init__(self, input_size, meal_type, config_path):
        self.config = load_config(config_path)
        self.input_size = input_size
        PresenceModel.kl_weight = K.variable(0.0)
        self.model = self.build_model()
        self.model_save_path = construct_save_path(meal_type, 'presence_model', 'model', self.config) 
        self.results_save_path = construct_save_path(meal_type, 'presence_model', 'results', self.config) 

    def build_model(self):
        encoder_inputs = Input(shape=(self.input_size,))
        encoder = encoder_inputs
        for size in self.config["presence_model"]["encoder_sizes"]:
            encoder = Dense(size, activation='relu')(encoder)
            encoder = Dropout(self.config["presence_model"]["dropout_rate"])(encoder)
            
        z_mean = Dense(self.config["presence_model"]["latent_dim"], name='z_mean')(encoder)
        z_log_var = Dense(self.config["presence_model"]["latent_dim"], name='z_log_var')(encoder)
        z = Lambda(sampling, output_shape=(self.config["presence_model"]["latent_dim"],), name='z')([z_mean, z_log_var])

        decoder = z
        for i, size in enumerate(self.config["presence_model"]["decoder_sizes"]):
            decoder = Dense(size, activation='relu', name=f'decoder_dense_{i}')(decoder)
        ingredient_selection = Dense(self.input_size, activation='sigmoid', name='food_selection_output')(decoder)

        vae_output = VAELossLayer(self.config)([ingredient_selection, z_mean, z_log_var, encoder_inputs])
        vae = Model(inputs=encoder_inputs, outputs=vae_output)
        vae.compile(optimizer=Adam(self.config["presence_model"]["learning_rate"]))
        return vae

    """
    def build_model(self):
        encoder_inputs = Input(shape=(self.input_size,))
        encoder = encoder_inputs
        for size in self.config["presence_model"]["encoder_sizes"]:
            encoder = Dense(size, activation='relu')(encoder)
            encoder = Dropout(self.config["presence_model"]["dropout_rate"])(encoder)
            
        z_mean = Dense(self.config["presence_model"]["latent_dim"], name='z_mean')(encoder)
        z_log_var = Dense(self.config["presence_model"]["latent_dim"], name='z_log_var')(encoder)
        z = Lambda(sampling, output_shape=(self.config["presence_model"]["latent_dim"],), name='z')([z_mean, z_log_var])

        decoder = z
        for i, size in enumerate(self.config["presence_model"]["decoder_sizes"]):
            decoder = Dense(size, activation='relu', name=f'decoder_dense_{i}')(decoder)
        ingredient_selection = Dense(self.input_size, activation='sigmoid', name='food_selection_output')(decoder)

        # Use VAELossLayer with the updated loss including custom metrics
        vae_output = VAELossLayer(self.config)([ingredient_selection, z_mean, z_log_var, encoder_inputs])
        vae = Model(inputs=encoder_inputs, outputs=vae_output)
        vae.compile(optimizer=Adam(self.config["presence_model"]["learning_rate"]))
        return vae   
    """
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
            verbose=1
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

    def extract_decoder(self):
        latent_inputs = Input(shape=(self.config["presence_model"]["latent_dim"],), name='z_sampling')
        x = latent_inputs
        for i, _ in enumerate(self.config["presence_model"]["decoder_sizes"]):
            layer = self.model.get_layer(f'decoder_dense_{i}')
            x = layer(x)
        output_layer = self.model.get_layer('food_selection_output')
        decoder_output = output_layer(x)
        
        return Model(latent_inputs, decoder_output)
        
    def save_models(self, log):
        self.model.save(self.model_save_path + 'vae_model.keras')
        decoder = self.extract_decoder()
        decoder.save(self.model_save_path + 'decoder_model.keras')
        if log is not None:
            log.info(f"Models saved to {self.model_save_path}vae_model.keras and {self.model_save_path}decoder_model.keras")
    
    @staticmethod
    def load_model(filepath):
        custom_objects = {'VAELossLayer': VAELossLayer}
        return load_model(filepath, custom_objects=custom_objects, safe_mode=False)
        
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

def rmse_loss(y_true, y_pred):
    return K.sqrt(K.mean(K.square(y_pred - y_true)))

def r_squared(y_true, y_pred):
    ss_res = K.sum(K.square(y_true - y_pred))
    ss_tot = K.sum(K.square(y_true - K.mean(y_true)))
    return (1 - ss_res/(ss_tot + K.epsilon()))

def sampling(args):
    z_mean, z_log_var = args
    batch = K.shape(z_mean)[0]
    dim = K.int_shape(z_mean)[1]
    epsilon = K.random_normal(shape=(batch, dim), mean=0., stddev=0.5)
    return z_mean + K.exp(0.5 * z_log_var) * epsilon

def food_coverage_loss(y_pred):
    """Encourage the model to activate a broader set of outputs."""
    epsilon = K.epsilon()
    mean_activation = K.mean(y_pred, axis=0)  # Average activation per output across the batch
    coverage = -K.mean(K.log(mean_activation + epsilon))  # Maximize mean activation, minimize the negative log
    return coverage

def gini_coefficient(y_true, y_pred):
    # Note: y_pred is the predicted distribution of ingredients.
    # Sorting the predictions
    values = tf.sort(y_pred, axis=1)
    n = tf.cast(tf.shape(values)[1], tf.float32)
    index = tf.range(1, n+1, dtype=tf.float32)
    
    # Calculating the Gini coefficient using the formula
    numerator = tf.reduce_sum((2 * index - n - 1) * values, axis=1)
    denominator = tf.reduce_sum(values, axis=1)
    gini = numerator / (n * denominator)
    # Since we want to minimize the loss, and a lower Gini coefficient indicates more equality
    # (which is our goal if we want to maximize diversity), we return 1 minus the Gini coefficient.
    return 1 - gini  # Minimizing inequality (maximizing equality)

def jaccard_index_loss(y_true, y_pred):
    """Approximate Jaccard Index for differentiable loss calculation."""
    epsilon = K.epsilon()
    y_true_ = K.clip(y_true, epsilon, 1)
    y_pred_ = K.clip(y_pred, epsilon, 1)
    intersection = K.sum(y_true_ * y_pred_, axis=-1)
    sum_ = K.sum(y_true_ + y_pred_, axis=-1)
    union = sum_ - intersection
    return 1 - (intersection / (union + epsilon))
