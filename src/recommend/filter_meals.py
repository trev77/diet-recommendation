import os
import sys
import numpy as np
import pandas as pd

from sklearn.manifold import TSNE
from sklearn.cluster import DBSCAN
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import pairwise_distances

import matplotlib.pyplot as plt
from matplotlib.ticker import NullFormatter
import seaborn as sns

import plotly.express as px
import plotly.graph_objects as go

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.helpers import *
from recommend.generate_meals import *

def evaluate_and_visualize_embeddings(real_embeddings, generated_embeddings, visualize):
    # Ensure 'node' or meal names are columns and not indices
    if real_embeddings.index.name == 'node':
        real_embeddings = real_embeddings.reset_index()
    if generated_embeddings.index.name == 'node':
        generated_embeddings = generated_embeddings.reset_index()

    # Standardize features of real embeddings
    scaler = MinMaxScaler()
    real_scaled_embeddings = scaler.fit_transform(real_embeddings.iloc[:, 1:])
    
    if not os.path.exists('practical_generated.pkl'):
        # Heuristic for DBSCAN's eps
        min_samples = 2 * real_scaled_embeddings.shape[1]
        nearest_neighbors = NearestNeighbors(n_neighbors=min_samples)
        nearest_neighbors.fit(real_scaled_embeddings)
        distances, indices = nearest_neighbors.kneighbors(real_scaled_embeddings)

        min_samples = int(min_samples/3)
        print(min_samples)
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
                
        to_pickle(practical_generated, 'practical_generated.pkl')
        to_pickle(impractical_generated, 'impractical_generated.pkl')
        to_pickle(real_clusters, 'real_clusters.pkl')
    else:
        practical_generated = pd.read_pickle('practical_generated.pkl')
        impractical_generated = pd.read_pickle('impractical_generated.pkl')
        real_clusters = pd.read_pickle('real_clusters.pkl')
    
    if visualize is True:
        # Visualization with t-SNE
        combined_scaled_embeddings = np.vstack([real_embeddings.iloc[:, 1:].to_numpy(), generated_embeddings.iloc[:, 1:].to_numpy()])
        # t-SNE for dimensionality reduction
        if not os.path.exists('tsne_embeddings.pkl'):
            tsne = TSNE(n_components=2, perplexity=50, n_iter=1500, random_state=42)
            tsne_embeddings = tsne.fit_transform(combined_scaled_embeddings)
            to_pickle(tsne_embeddings, 'tsne_embeddings.pkl')
        else:
            tsne_embeddings = pd.read_pickle('tsne_embeddings.pkl')
        
        plot_figure(real_clusters, tsne_embeddings, generated_embeddings, practical_generated, real_scaled_embeddings, impractical_generated)
        plot_plotly_fig(real_clusters, tsne_embeddings, generated_embeddings, practical_generated, real_scaled_embeddings, impractical_generated)
        
    return practical_generated, impractical_generated

from matplotlib.patches import ConnectionPatch
"""
def plot_figure(real_clusters, tsne_embeddings, generated_embeddings, practical_generated, real_scaled_embeddings, impractical_generated):
    palette = sns.color_palette("tab20c", 10).as_hex()
    palette2 = sns.color_palette("tab20b", 10).as_hex()
    # Assuming tsne_embeddings, real_clusters, practical_generated, impractical_generated are defined
    # Filter out the real meals that are labeled as outliers by DBSCAN
    non_outlier_indices = np.where(real_clusters != -1)[0]
    # Main plot
    fig, ax = plt.subplots(figsize=(12, 6))
    main_point_size = 16
    # Plot non-outlier real meals in main plot
    ax.scatter(tsne_embeddings[non_outlier_indices, 0], tsne_embeddings[non_outlier_indices, 1], s=main_point_size, c=palette2[1], edgecolors=palette2[0], label='Real Meals')
    # Plot practical generated meals in main plot
    practical_indices = [index for index, label in enumerate(generated_embeddings.iloc[:, 0]) if label in practical_generated]
    ax.scatter(tsne_embeddings[len(real_scaled_embeddings):][practical_indices, 0], tsne_embeddings[len(real_scaled_embeddings):][practical_indices, 1], s=main_point_size+20, c=palette[9], edgecolors=palette[8],label='Practical Generated Meals')
    # Plot impractical generated meals in main plot
    impractical_indices = [index for index, label in enumerate(generated_embeddings.iloc[:, 0]) if label in impractical_generated]
    ax.scatter(tsne_embeddings[len(real_scaled_embeddings):][impractical_indices, 0], tsne_embeddings[len(real_scaled_embeddings):][impractical_indices, 1], s=main_point_size+20, c=palette[5], edgecolors=palette[4], label='Impractical Generated Meals')
    ax.set_xlim([30,60])
    ax.set_ylim([-68, -40])
    ax.set_title('t-SNE Projection of Meal Embeddings')
    ax.set_xlabel('t-SNE 1')
    ax.set_ylabel('t-SNE 2')
    ax.legend()

    # Inset plot
    inset_left = 0.6  # 65% from the left of the figure
    inset_bottom = 0.58  # 70% from the bottom of the figure
    inset_width = 0.3  # 30% of the figure's width
    inset_height = 0.3  # 30% of the figure's height
    ax_inset = fig.add_axes([inset_left, inset_bottom, inset_width, inset_height])
    inset_point_size = 0.5
    ax_inset.scatter(tsne_embeddings[non_outlier_indices, 0], tsne_embeddings[non_outlier_indices, 1], s=inset_point_size, c=palette2[1], edgecolors=palette2[0])
    ax_inset.scatter(tsne_embeddings[len(real_scaled_embeddings):][practical_indices, 0], tsne_embeddings[len(real_scaled_embeddings):][practical_indices, 1], s=inset_point_size, c=palette[9], edgecolors=palette[8])
    ax_inset.scatter(tsne_embeddings[len(real_scaled_embeddings):][impractical_indices, 0], tsne_embeddings[len(real_scaled_embeddings):][impractical_indices, 1], s=inset_point_size, c=palette[5], edgecolors=palette[4])

    # Set limits for the inset plot
    ax_inset.set_xticklabels([])
    ax_inset.set_yticklabels([])
    ax_inset.xaxis.set_major_formatter(NullFormatter())
    ax_inset.yaxis.set_major_formatter(NullFormatter())
    
    plt.savefig('meal_embeddings_visualization.png', dpi=500)
"""

def plot_figure(real_clusters, tsne_embeddings, generated_embeddings, practical_generated, real_scaled_embeddings, impractical_generated):
    palette = sns.color_palette("tab20c", 10).as_hex()
    palette2 = sns.color_palette("tab20b", 10).as_hex()
    main_point_size = 16
    inset_point_size = 2

    fig, ax_main = plt.subplots(figsize=(15, 10))

    # Define positions and sizes for the inset axes
    inset_specs = [
        {'pos': [0.65, 0.05, 0.3, 0.3], 'zoom_area': [30, -68, 30, 28]},  # bottom right
        {'pos': [0.65, 0.65, 0.3, 0.3], 'zoom_area': [-50, -10, 20, 20]},  # top right
        {'pos': [0.05, 0.65, 0.3, 0.3], 'zoom_area': [-30, 40, 30, 20]}  # top left
    ]

    for spec in inset_specs:
        ax_inset = fig.add_axes(spec['pos'])  # left, bottom, width, height relative to figure size
        ax_inset.scatter(tsne_embeddings[:, 0], tsne_embeddings[:, 1], s=20, c=palette2[0], alpha=0.5)
        ax_inset.set_xlim(spec['zoom_area'][0], spec['zoom_area'][0] + spec['zoom_area'][2])
        ax_inset.set_ylim(spec['zoom_area'][1], spec['zoom_area'][1] + spec['zoom_area'][3])
        ax_inset.set_xticklabels([])
        ax_inset.set_yticklabels([])
        ax_inset.xaxis.set_major_formatter(NullFormatter())
        ax_inset.yaxis.set_major_formatter(NullFormatter())

        # Draw rectangle on main plot
        ax_main.add_patch(plt.Rectangle((spec['zoom_area'][0], spec['zoom_area'][1]), spec['zoom_area'][2], spec['zoom_area'][3], fill=False, edgecolor='black', linestyle='--', linewidth=2))
        # Main Plot (now used as the inset for detailed view)
        non_outlier_indices = np.where(real_clusters != -1)[0]
        practical_indices = [index for index, label in enumerate(generated_embeddings.iloc[:, 0]) if label in practical_generated]
        impractical_indices = [index for index, label in enumerate(generated_embeddings.iloc[:, 0]) if label in impractical_generated]
        ax_inset.scatter(tsne_embeddings[non_outlier_indices, 0], tsne_embeddings[non_outlier_indices, 1], s=30, c=palette2[1], label='Real Meals (Non-outlier)')
        ax_inset.scatter(tsne_embeddings[len(real_scaled_embeddings):][practical_indices, 0], tsne_embeddings[len(real_scaled_embeddings):][practical_indices, 1], s=main_point_size+20, c=palette[9], edgecolors=palette[8],label='Practical Generated Meals')
        ax_inset.scatter(tsne_embeddings[len(real_scaled_embeddings):][impractical_indices, 0], tsne_embeddings[len(real_scaled_embeddings):][impractical_indices, 1], s=main_point_size+20, c=palette[5], edgecolors=palette[4], label='Impractical Generated Meals')
        # Draw lines connecting the main plot rectangle to the inset, adjusted for the new position
        ax_main.add_patch(plt.Rectangle((spec['zoom_area'][0], spec['zoom_area'][1]), spec['zoom_area'][2], spec['zoom_area'][3], fill=False, edgecolor='red', linestyle='--', linewidth=2))

        coordsA = "data"
        coordsB = "axes fraction"
        con1 = ConnectionPatch(xyA=(spec['zoom_area'][0], spec['zoom_area'][1] + spec['zoom_area'][3]), xyB=(0, 1), coordsA=coordsA, coordsB=coordsB, axesA=ax_main, axesB=ax_inset, color="red", linestyle="--")
        con2 = ConnectionPatch(xyA=(spec['zoom_area'][0] + spec['zoom_area'][2], spec['zoom_area'][1] + spec['zoom_area'][3]), xyB=(1, 1), coordsA=coordsA, coordsB=coordsB, axesA=ax_main, axesB=ax_inset, color="red", linestyle="--")
        fig.add_artist(con1)
        fig.add_artist(con2)

    # Main Plot (acting as the overview)
    ax_main.scatter(tsne_embeddings[:, 0], tsne_embeddings[:, 1], s=20, c=palette2[0], alpha=0.5)
    # Main Plot (now used as the inset for detailed view)
    non_outlier_indices = np.where(real_clusters != -1)[0]
    practical_indices = [index for index, label in enumerate(generated_embeddings.iloc[:, 0]) if label in practical_generated]
    impractical_indices = [index for index, label in enumerate(generated_embeddings.iloc[:, 0]) if label in impractical_generated]
    ax_main.scatter(tsne_embeddings[non_outlier_indices, 0], tsne_embeddings[non_outlier_indices, 1], s=inset_point_size, c=palette2[1], edgecolors=palette2[0])
    ax_main.scatter(tsne_embeddings[len(real_scaled_embeddings):][practical_indices, 0], tsne_embeddings[len(real_scaled_embeddings):][practical_indices, 1], s=inset_point_size, c=palette[9], edgecolors=palette[8])
    ax_main.scatter(tsne_embeddings[len(real_scaled_embeddings):][impractical_indices, 0], tsne_embeddings[len(real_scaled_embeddings):][impractical_indices, 1], s=inset_point_size, c=palette[5], edgecolors=palette[4])
    ax_main.set_title('Overview of Meal Embeddings')
    ax_main.set_xlabel('t-SNE 1')
    ax_main.set_ylabel('t-SNE 2')

    plt.savefig('meal_embeddings_visualization_multi_inset.png', dpi=500)
    

def plot_plotly_fig(real_clusters, tsne_embeddings, generated_embeddings, practical_generated, real_scaled_embeddings, impractical_generated):
    # Unique clusters including -1 for outliers
    unique_clusters = np.unique(real_clusters)
    print(len(unique_clusters))
    colors = px.colors.qualitative.Plotly  # Using Plotly's qualitative color scale
    # Assign a color to each cluster
    cluster_colors = {cluster: colors[i % len(colors)] for i, cluster in enumerate(unique_clusters)}
    # Assign a special color for outliers (e.g., black)
    cluster_colors[-1] = 'black'
    # Start building the Plotly figure
    fig = go.Figure()
    # Plot for real meals in clusters
    for cluster in unique_clusters:
        cluster_indices = np.where(real_clusters == cluster)[0]
        fig.add_trace(go.Scatter(
            x=tsne_embeddings[cluster_indices, 0],
            y=tsne_embeddings[cluster_indices, 1],
            mode='markers',
            marker=dict(color=cluster_colors[cluster], size=5),
            name=f'Real Meals - Cluster {cluster}'
        ))
    # Plot for practical generated meals
    practical_indices = [index for index, label in enumerate(generated_embeddings.iloc[:, 0]) if label in practical_generated]
    fig.add_trace(go.Scatter(
        x=tsne_embeddings[len(real_scaled_embeddings):][practical_indices, 0],
        y=tsne_embeddings[len(real_scaled_embeddings):][practical_indices, 1],
        mode='markers',
        marker=dict(color='green', size=5, symbol='circle'),
        name='Practical Generated Meals'
    ))
    # Plot for impractical generated meals
    impractical_indices = [index for index, label in enumerate(generated_embeddings.iloc[:, 0]) if label in impractical_generated]
    fig.add_trace(go.Scatter(
        x=tsne_embeddings[len(real_scaled_embeddings):][impractical_indices, 0],
        y=tsne_embeddings[len(real_scaled_embeddings):][impractical_indices, 1],
        mode='markers',
        marker=dict(color='red', size=5, symbol='x'),
        name='Impractical Generated Meals'
    ))
    # Update layout
    fig.update_layout(
        title='t-SNE Projection with Cluster and Practicality Coloring',
        xaxis_title='t-SNE 1',
        yaxis_title='t-SNE 2',
        legend_title='Legend',
        margin=dict(l=0, r=0, t=30, b=0)
    )
    fig.write_html('plot.html')
    
def main():
    real_embeddings = pd.read_pickle('results/embedding_model/real_meal_embeddings.pkl')
    generated_embeddings = pd.read_pickle('results/embedding_model/real_and_generated_meal_embeddings.pkl')

    real_meals = real_embeddings[real_embeddings['node'].str.contains('real')]
    generated_meals = generated_embeddings[generated_embeddings['node'].str.contains('generated')]
    real_meals = generated_embeddings[generated_embeddings['node'].str.contains('real')]
    practical_generated, impractical_generated = evaluate_and_visualize_embeddings(real_meals, generated_meals, True)
    to_pickle(impractical_generated, 'results/embedding_model/impractical_generated_meals.pkl')
    to_pickle(practical_generated, 'results/embedding_model/practical_generated_meals.pkl')


if __name__ == '__main__':
    main()




