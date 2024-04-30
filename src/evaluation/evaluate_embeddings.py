import os
from tqdm import tqdm
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, roc_curve, precision_recall_curve, auc, f1_score
from sklearn.metrics.pairwise import cosine_similarity

def load_graph(graph_file):
    try:
        graph_df = pd.read_csv(graph_file, header=None, names=['node1', 'node2', 'weight'])
    except FileNotFoundError:
        raise Exception(f"The file {graph_file} was not found.")
    return graph_df[graph_df['weight'] > 0]

def get_similarity_matrix(embeddings_df):
    embeddings_matrix = embeddings_df.iloc[:, 1:].values
    similarity_matrix = cosine_similarity(embeddings_matrix)
    return similarity_matrix

def calculate_similarities(graph_df, embeddings_df, similarity_matrix):
    node_to_index = {node: idx for idx, node in enumerate(embeddings_df['node'])}
    similarities = []
    for _, row in tqdm(graph_df.iterrows(), total=graph_df.shape[0], desc="Calculating Similarities"):
        similarity = similarity_matrix[node_to_index[row['node1']], node_to_index[row['node2']]]
        similarities.append(similarity)
    return pd.Series(similarities)

def shuffle_and_calculate_similarities(graph_df, embeddings_df, similarity_matrix):
    shuffled_df = graph_df.copy()
    shuffled_df['node2'] = np.random.permutation(shuffled_df['node2'].values)
    node_to_index = {node: idx for idx, node in enumerate(embeddings_df['node'])}
    shuffled_similarities = []
    for _, row in tqdm(shuffled_df.iterrows(), total=shuffled_df.shape[0], desc="Calculating Shuffled Similarities"):
        similarity = similarity_matrix[node_to_index[row['node1']], node_to_index[row['node2']]]
        shuffled_similarities.append(similarity)
    return pd.Series(shuffled_similarities)

def evaluate_embeddings(true_similarities, shuffled_similarities):
    labels = np.concatenate([np.ones(len(true_similarities)), np.zeros(len(shuffled_similarities))])
    preds = np.concatenate([true_similarities, shuffled_similarities])
    preds_bin = np.array([1 if pred > 0.5 else 0 for pred in preds])
    
    fpr, tpr, _ = roc_curve(labels, preds)
    precision, recall, _ = precision_recall_curve(labels, preds)
    auc_roc = auc(fpr, tpr)
    auc_pr = auc(recall, precision)
    f1 = f1_score(labels, preds_bin)

    return fpr, tpr, recall, precision, auc_roc, auc_pr, f1

def plot_curves(fpr, tpr, recall, precision, auc_roc, auc_pr, file_prefix, embedding_size, algo):
    plot_file_prefix = f"{file_prefix.split('/')[-1].split('.')[0]}_{embedding_size}_{algo}"
    outdir = 'results/embedding_model/'
    os.makedirs(outdir, exist_ok=True)

    plt.figure(figsize=(8,8))
    plt.plot(fpr, tpr, label=f'AUC ROC = {auc_roc:.2f}')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC Curve: {plot_file_prefix}')
    plt.legend(loc='lower right')
    plt.savefig(f'{outdir}/{plot_file_prefix}_roc.jpg', dpi=1000)
    plt.close()

    plt.figure(figsize=(8,8))
    plt.plot(recall, precision, label=f'AUC PR = {auc_pr:.2f}')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title(f'Precision-Recall Curve: {plot_file_prefix}')
    plt.legend(loc='upper right')
    plt.savefig(f'{outdir}/{plot_file_prefix}_pr.jpg', dpi=1000)
    plt.close()

def eval_embeddings(graph_file, embeddings_df, embedding_size, algo, log):
    graph_df = load_graph(graph_file)
    log.info(f"Evaluating graph embeddings...")
    similarity_matrix = get_similarity_matrix(embeddings_df)

    true_similarities = calculate_similarities(graph_df, embeddings_df, similarity_matrix)
    shuffled_similarities = shuffle_and_calculate_similarities(graph_df, embeddings_df, similarity_matrix)
    fpr, tpr, recall, precision, auc_roc, auc_pr, f1 = evaluate_embeddings(true_similarities, shuffled_similarities)
    plot_curves(fpr, tpr, recall, precision, auc_roc, auc_pr, graph_file, embedding_size, algo)

    log.info(f"Graph reconstruction F1: {f1}")
    log.info(f"AUROC of network reconstruction task: {auc_roc}")
    log.info(f"AUPR of network reconstruction task: {auc_pr}")
