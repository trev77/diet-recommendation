import pandas as pd

def calculate_meal_metrics(meal_type):
    from scipy.stats import entropy
    from math import log2

    generated_meals = pd.read_pickle('generated_meals/breakfast_practical_meals.pkl')
    df_generated = meals_to_dataframe(generated_meals)
    df_real = pd.read_csv('data/{}_preprocessed.csv'.format(meal_type), index_col='Unnamed: 0')

    # 1. Find unique foods in generated meals
    unique_foods_generated = (df_generated.columns[df_generated.sum(axis=0) > 0]).tolist()

    # 2. Find unique foods that are possible (based on real dataset in this example)
    # If you have a predetermined list of possible foods, use that list instead of df_real.columns.tolist()
    unique_foods_possible = df_real.columns.tolist()

    # 3. Calculate food coverage
    food_coverage = len(set(unique_foods_generated)) / len(set(unique_foods_possible))
    print("Food Coverage:", food_coverage)

    # Calculate meal coverage
    # Convert meals to strings to make them hashable
    unique_meals = len(set([str(x) for x in (df_generated > 0).values.tolist()]))
    all_possible_meals = 2 ** df_generated.shape[1] - 1 # 2 to the power of the number of foods
    meal_coverage = unique_meals / all_possible_meals
    print("Meal Coverage:", meal_coverage)

    # Calculate meal diversity (entropy of generated meals)
    meal_freqs = df_generated.mean(axis=0) / df_generated.mean(axis=0).sum()  # Frequency of each food in generated meals
    meal_diversity = entropy(meal_freqs, base=2)  # Calculate entropy
    print("Meal Diversity:", meal_diversity)

    # Calculate meal realism (KL divergence from real to generated meals)
    # Determine union of foods
    all_foods = set(df_real.columns) | set(df_generated.columns)

    # Expand real_freqs and meal_freqs to match this union set
    expanded_real_freqs = [df_real[food].mean() if food in df_real.columns else 1e-10 for food in all_foods]
    expanded_generated_freqs = [df_generated[food].mean() if food in df_generated.columns else 1e-10 for food in all_foods]

    # Normalize the frequencies so they sum up to 1
    total_real = sum(expanded_real_freqs)
    total_generated = sum(expanded_generated_freqs)

    expanded_real_freqs = [freq / total_real for freq in expanded_real_freqs]
    expanded_generated_freqs = [freq / total_generated for freq in expanded_generated_freqs]

    # Add a small constant to avoid division by zero and log of zero
    epsilon = 1e-10
    expanded_real_freqs = [freq + epsilon for freq in expanded_real_freqs]
    expanded_generated_freqs = [freq + epsilon for freq in expanded_generated_freqs]

    # Calculate KL-divergence
    meal_realism = entropy(expanded_real_freqs, expanded_generated_freqs, base=2)
    print("Meal Realism (KL-Divergence):", meal_realism)

def filter_generated_meals(meal_type, embedding_df, meal_obj_lookup, metric='cosine'):
    # Save labels and names before dropping them
    labels = embedding_df['labels'].copy()
    names = embedding_df['names'].copy()
    descrip = embedding_df['descrip'].copy()

    perp = 50
    lr = 10
    embedding_size = 32

    # Run t-SNE on the original embeddings
    tsne_df = run_tsne(embedding_df.drop(['labels', 'names', 'descrip'], axis=1), embedding_size, lr, perp)
    tsne_df.index = embedding_df.index
    tsne_df['labels'] = embedding_df['labels']
    tsne_df['names'] = embedding_df['names']
    tsne_df['descrip'] = embedding_df['descrip']

    graph_file = 'real_gen_graph'
    plot_tsne(meal_type, tsne_df, lr, perp, embedding_size, graph_file)
    
    print(tsne_df)
    
    # Separate the generated and real meals
    generated_meals = tsne_df[tsne_df['labels'] == 'generated']
    real_meals = tsne_df[tsne_df['labels'] == 'real']

    # Separate the generated and real meals
    #generated_meals = embedding_df[embedding_df['labels'] == 'generated']
    #real_meals = embedding_df[embedding_df['labels'] == 'real']

    embedding_df =  tsne_df.drop(['labels', 'names', 'descrip'], axis=1)
    #embedding_df = embedding_df.drop(['labels', 'names', 'descrip'], axis=1)

    # Initialize the best parameters and metrics
    best_n_neighbors = None
    best_threshold = None
    best_nof = float('-inf')
    best_disparity = 0

    n_neighbors_range = [15]
    threshold_range = [-2]
    for n_neighbors in n_neighbors_range:
        print('Neighbor loop: {}'.format(n_neighbors))
        lof = LocalOutlierFactor(n_neighbors=n_neighbors, contamination='auto')
        lof.fit(embedding_df)
        nof = lof.negative_outlier_factor_
        print(min(nof))
        print(max(nof))
        for threshold in threshold_range:
            print('\tThreshold: {}'.format(threshold))
            thresholded_nof = nof < threshold

            # Find the outlier/inlier indices for the generated meals
            outlier_indices = embedding_df[thresholded_nof].index.intersection(generated_meals.index)
            inlier_indices = embedding_df[~thresholded_nof].index.intersection(generated_meals.index)

            # Filter the outliers/inliers from embedding_df using the indices
            filtered_meals = embedding_df.loc[outlier_indices]
            kept_meals = embedding_df.loc[inlier_indices]

            # Compute distances only once
            distances_filtered = pairwise_distances(filtered_meals, real_meals.drop(['labels', 'names', 'descrip'], axis=1), metric=metric)
            distances_kept = pairwise_distances(kept_meals, real_meals.drop(['labels', 'names', 'descrip'], axis=1), metric=metric)
            avg_distance = np.mean(np.concatenate((distances_filtered, distances_kept), axis=None))

            sim_filtered = distances_filtered.mean()
            sim_kept = distances_kept.mean()

            # Calculate the disparity between kept and filtered meals
            disparity = abs(sim_filtered - sim_kept)

            # Update the best parameters based on the maximum disparity
            if disparity > best_disparity:
                best_disparity = disparity
                best_n_neighbors = n_neighbors
                best_threshold = threshold

    print('Best disparity: {}'.format(best_disparity))
    print('Best number of neighbors: {}'.format(best_n_neighbors))
    print('Best threshold: {}'.format(best_threshold))

    # Perform LOF analysis on the entire dataframe using the best parameters
    lof = LocalOutlierFactor(n_neighbors=best_n_neighbors, contamination='auto')
    lof.fit(embedding_df)
    nof = lof.negative_outlier_factor_

    filtered_indices_final = generated_meals.index.isin(embedding_df[nof < best_threshold].index)
    kept_indices_final = generated_meals.index.isin(embedding_df[nof >= best_threshold].index)

    filtered_meal_embeddings = generated_meals[filtered_indices_final]
    kept_meal_embeddings = generated_meals[kept_indices_final]

    filtered_meals_final = [value for key, value in meal_obj_lookup.items() if key in list(filtered_meal_embeddings['names'])]
    kept_meals_final = [value for key, value in meal_obj_lookup.items() if key in list(kept_meal_embeddings['names'])]

    return filtered_meals_final, kept_meals_final, filtered_indices_final, kept_indices_final




def eval_embeddings(graph_file, embeddings_df, embedding_size, algo):
    """Evaluate the quality of node embeddings learned by a given algorithm using a graph represented by an edge list stored in a CSV file.

    Parameters
    ----------
    - graph_file (str):
        The name of the CSV file containing the graph edge list
    - embeddings_df (Pandas dataframe):
        A DataFrame containing the node embeddings
    - embedding_size (int):
        The size of the node embeddings
    - algo (str):
        The name of the algorithm used to learn the embeddings

    Returns
    -------
    - fpr (list):
        false positive rate
    - tpr (list):
        true positive rate
    - auc_score (float):
        AUC score of ROC curve
    """

    # Load the graph from the CSV file
    graph_df = pd.read_csv(graph_file, header=None, names=['node1', 'node2', 'weight'])
    graph_df = graph_df[graph_df['weight'] > 0] # Only keep edges with non-zero weight

    # Create a dictionary to store the true similarities for each edge in the graph
    true_similarities = {}
    for _, row in graph_df.iterrows():
        true_similarities[(row['node1'], row['node2'])] = embedding_cosine_sim(embeddings_df, row['node1'], row['node2'])

    # Shuffle the edges of the graph
    shuffled_df = graph_df
    shuffled_df['node1'] = graph_df['node1']
    shuffled_df['node2'] = graph_df['node2'].sample(frac=1, random_state=42).values

    # Create a dictionary to store the shuffled similarities for each edge
    shuffled_similarities = {}
    for _, row in shuffled_df.iterrows():
        shuffled_similarities[(row['node1'], row['node2'])] = embedding_cosine_sim(embeddings_df, row['node1'], row['node2'])

    # Compute the AUROC between the true similarities and the shuffled similarities
    labels = [1] * len(true_similarities) + [0] * len(shuffled_similarities)
    preds = list(true_similarities.values()) + list(shuffled_similarities.values())
    preds_bin = [1 if pred > 0.5 else 0 for pred in preds]
    auc_roc = roc_auc_score(labels, preds)

    # Compute the precision, recall, and auc of the PR curve
    precision, recall, _ = precision_recall_curve(labels, preds)
    auc_pr = auc(recall, precision)
    f1 = f1_score(labels, preds_bin)

    # # Compute the false positive rate, true positive rate, and thresholds for the ROC curve
    fpr, tpr, thresholds = roc_curve(labels, preds)
    auc_roc = auc(fpr, tpr)

    # Plot the ROC curve
    plt.figure(figsize=(5,5), dpi=1200)
    plt.plot(fpr, tpr, label='AUC = %0.2f' % auc_roc)
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('False positive rate')
    plt.ylabel('True positive rate')
    plt.legend(loc='lower right')
    plt.title('{} Embeddings - embedding size {}'.format(graph_file, embedding_size))
    plt.savefig(
        'embeddings/results/graphs/{}_{}_{}_roc.jpg'.format(graph_file.split('/')[-1].split('.')[0], embedding_size, algo),
        pil_kwargs={
            'quality': 100,
            'subsampling': 10
        })
    plt.show()

    # Plot the precision-recall curve
    plt.figure(figsize=(5,5), dpi=1200)
    plt.plot(recall, precision, label='AUC = %0.2f' % auc_pr)

    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.legend(loc='upper right')
    plt.title('{} Embeddings - embedding size {}'.format(graph_file, embedding_size))
    plt.savefig(
        'embeddings/results/graphs/{}_{}_{}_pr.jpg'.format(graph_file.split('/')[-1].split('.')[0], embedding_size, algo),
        pil_kwargs={
            'quality': 100,
            'subsampling': 10
        })
    plt.show()

    print("Graph reconstruction F1: {}".format(f1))
    print("AUROC of network reconstruction task: {}".format(auc_roc))
    print("AUPR of network reconstruction task: {}".format(auc_pr))
    eval_stats = multilabel_metrics(labels, preds_bin)
    
    
    
def filter_generated_meals(embedding_df, meal_obj_lookup, metric='cosine'):
    # Separate the generated and real meals
    generated_meals = embedding_df[embedding_df['labels'] == 'generated']
    real_meals = embedding_df[embedding_df['labels'] == 'real']
    
    # Drop the labels and names columns for LOF analysis
    embedding_df = embedding_df.drop(['labels', 'names'], axis=1)
    
    # Perform LOF analysis on the entire dataframe and calculate the negative outlier factor
    best_n_neighbors = None
    best_threshold = None
    best_nof = float('-inf')
    best_similarity = 0
    best_disparity = 0
    
    n_neighbors_range = [20, 30]
    threshold_range = [-2.0, -1.5, -1.0]
    for n_neighbors in range(n_neighbors_range[0], n_neighbors_range[1]+1):
        lof = LocalOutlierFactor(n_neighbors=n_neighbors, contamination='auto')
        lof.fit(embedding_df)
        nof = lof.negative_outlier_factor_
        for threshold in threshold_range:
            thresholded_nof = nof < threshold
            filtered_meals = embedding_df[thresholded_nof][embedding_df[thresholded_nof].index.isin(generated_meals.index)]
            kept_meals = embedding_df[~thresholded_nof][embedding_df[~thresholded_nof].index.isin(generated_meals.index)]

            # Calculate the average distance between kept and filtered meals to the realistic meal embeddings
            distances = pairwise_distances(kept_meals, real_meals.drop(['labels', 'names'], axis=1), metric=metric)
            avg_distance = distances.mean()

            # Calculate the similarity between kept and filtered meals to the realistic meal embeddings
            sim_filtered = pairwise_distances(filtered_meals, real_meals.drop(['labels', 'names'], axis=1), metric=metric).mean()
            sim_kept = pairwise_distances(kept_meals, real_meals.drop(['labels', 'names'], axis=1), metric=metric).mean()
            # similarity = min(sim_filtered, sim_kept)

            # Calculate the disparity between kept and filtered meals
            disparity = abs(sim_filtered - sim_kept)

            # Update the best parameters based on the maximum disparity
            if disparity > best_disparity:
                # best_similarity = similarity
                best_disparity = disparity
                best_n_neighbors = n_neighbors
                best_threshold = threshold
    
    # Perform LOF analysis on the entire dataframe using the best parameters
    lof = LocalOutlierFactor(n_neighbors=best_n_neighbors, contamination='auto')
    lof.fit(embedding_df)
    nof = lof.negative_outlier_factor_
    keep_index = generated_meals.index.isin(embedding_df[nof < best_threshold][embedding_df[nof < best_threshold].index.isin(generated_meals.index)].index)
    filter_index = generated_meals.index.isin(embedding_df[nof >= best_threshold][embedding_df[nof >= best_threshold].index.isin(generated_meals.index)].index)
    filtered_meal_embeddings = generated_meals[filter_index]
    kept_meal_embeddings = generated_meals[keep_index]
    
    filtered_meals = [value for key, value in meal_obj_lookup.items() if key in list(filtered_meal_embeddings['names'])]
    kept_meals = [value for key, value in meal_obj_lookup.items() if key in list(kept_meal_embeddings['names'])]
    
    kept_meals_translated = []
    for meal in kept_meals:
        kept_meals_translated.append(decode_food_codes(meal.fdcd))
        
    filtered_meals_translated = []
    for meal in filtered_meals:
        filtered_meals_translated.append(decode_food_codes(meal.fdcd))
    
    # Display the anomaly scores for the generated meals
    # print(anomaly_scores)
    
    return filtered_meals, kept_meals, filter_index, keep_index