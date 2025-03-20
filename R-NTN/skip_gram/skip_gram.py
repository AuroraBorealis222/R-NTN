import csv
import pickle
from collections import defaultdict

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
from sklearn.manifold import TSNE
from tqdm import tqdm

from ge import Node2Vec


def plot_embeddings(embeddings):
    emb_list = [embeddings[node] for node in embeddings]
    emb_list = np.array(emb_list)
    model = TSNE(n_components=2)
    node_pos = model.fit_transform(emb_list)
    plt.scatter(node_pos[:, 0], node_pos[:, 1])
    plt.show()


import csv
import numpy as np
from collections import defaultdict
from tqdm import tqdm


def build_transaction_dicts(csv_file):
    transaction_volume = defaultdict(lambda: np.float64(0))
    transaction_volume_between = defaultdict(lambda: defaultdict(lambda: np.float64(0)))

    with open(csv_file, mode='r') as file:
        reader = csv.DictReader(file)
        rows = list(reader)
        for row in tqdm(rows, desc="Processing transactions"):
            source = row['Row']
            target = row['Column']
            amount = np.float64(float(row['Amount']))  # Ensure amount is float64
            transaction_volume[source] += amount
            transaction_volume[target] += amount
            transaction_volume_between[source][target] += amount
            transaction_volume_between[target][source] += amount

    # Convert defaultdict to dict with float64 format
    transaction_volume = {float(k): v for k, v in transaction_volume.items()}
    transaction_volume_between = {float(k): {float(kk): float(vv) for kk, vv in v.items()} for k, v in
                                  transaction_volume_between.items()}

    return transaction_volume, transaction_volume_between


def save_transaction_volume_to_csv(transaction_volume, output_file):
    # Convert defaultdict to a regular dictionary
    transaction_volume = dict(transaction_volume)

    # Sort the dictionary by keys (nodes) in ascending order
    sorted_items = sorted(transaction_volume.items(), key=lambda item: item[0])

    # Write the sorted data to a CSV file
    with open(output_file, mode='w', newline='') as file:
        writer = csv.writer(file)
        # Write header
        writer.writerow(['Node', 'Transaction Volume'])
        # Write sorted rows
        for node, volume in sorted_items:
            writer.writerow([node, volume])

    print(f"Data has been written to {output_file}")


def save_dict_to_file(data, filename):
    with open(filename, 'wb') as f:
        pickle.dump(data, f)


def load_dict_from_file(filename):
    with open(filename, 'rb') as f:
        data = pickle.load(f)
    return data


if __name__ == "__main__":
    with open('2-hop.pkl', 'rb') as file:
        G = pickle.load(file)

    
    G = nx.DiGraph(G)
    

    csv_file = 'edg.csv'
    transaction_volume, transaction_volume_between = build_transaction_dicts(csv_file)

    model = Node2Vec(G, transaction_volume, transaction_volume_between, walk_length=20, num_walks=50, workers=24)

    # Train the model and generate embeddings
    train_results = model.train(window=10, min_count=1)

    embeddings = model.get_embeddings()

    # Process embeddings data
    processed_data = [[str(node)] + list(embeddings[node]) for node in embeddings]

    # Save the processed data to a CSV file
    with open('embeddings_processed.csv', 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Node'] + [f'Embedding_{i + 1}' for i in range(len(embeddings[next(iter(embeddings))]))])
        writer.writerows(processed_data)
    # plot_embeddings(embeddings)
