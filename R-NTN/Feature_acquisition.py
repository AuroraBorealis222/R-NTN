import concurrent
import random
from concurrent.futures import ProcessPoolExecutor

import pandas as pd

from Feature_computation import *
from computation import computing


def get_isp1_nodes_and_random(g, attribute='isp'):
    isp1_nodes = [node for node, data in g.nodes(data=True) if data.get(attribute) == 1]
    all_nodes = list(g.nodes())
    if len(isp1_nodes) > len(all_nodes) / 2:
        raise ValueError(
            "The number of isp=1 nodes is more than half the total nodes, cannot select an equal number of random "
            "nodes.")
    random_nodes = random.sample([node for node in all_nodes if node not in isp1_nodes], len(isp1_nodes))
    combined_nodes = isp1_nodes + random_nodes
    return combined_nodes


def append_df_to_csv(df, csv_file):
    try:
        existing_df = pd.read_csv(csv_file)
        updated_df = pd.concat([existing_df, df], ignore_index=True)
    except FileNotFoundError:
        updated_df = df
    updated_df.to_csv(csv_file, index=False)


def process_node(node, G):
    df = computing(G, node)
    return df


def read_and_process_graph(graph_path, output_csv):
    with open(graph_path, 'rb') as f:
        G = pickle.load(f)
    nodes = get_isp1_nodes_and_random(G)
    batch_size = 5
    with ProcessPoolExecutor() as executor:
        for i in tqdm(range(0, len(nodes), batch_size), desc="Processing nodes"):
            current_batch = nodes[i:i + batch_size]
            futures = [executor.submit(process_node, node, G) for node in current_batch]
            batch_df = pd.DataFrame()
            for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures), desc="Processing batch"):
                batch_df = pd.concat([batch_df, future.result()], ignore_index=True)
            append_df_to_csv(batch_df, output_csv)


if __name__ == '__main__':
    graph_path = '2-hop.pkl'
    out_path = 'Global_feature.csv'
    read_and_process_graph(graph_path, out_path)
