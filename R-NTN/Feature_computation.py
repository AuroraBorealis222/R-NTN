import datetime
import math
import os
import pickle
import statistics
from collections import defaultdict
from datetime import datetime
from numbers import Number

import networkx as nx
import numpy as np
from networkx import NodeNotFound
from tqdm import tqdm


def get_node_transactions(G, node):
    if node not in G:
        return None, None
    input_amounts = []
    output_amounts = []
    for _, _, data in G.in_edges(node, data=True):
        input_amounts.append(data.get('amount', 0))
    for _, _, data in G.out_edges(node, data=True):
        output_amounts.append(data.get('amount', 0))

    return input_amounts, output_amounts


def calculate_metrics(g, node):
    edges = sorted(list(g.in_edges(node, data=True)) + list(g.out_edges(node, data=True)),
                   key=lambda x: x[2].get('timestamp', 0))
    daily_edges = defaultdict(list)
    for edge in edges:
        date = datetime.fromtimestamp(edge[2].get('timestamp', 0)).strftime('%Y-%m-%d')
        daily_edges[date].append(edge)
        if edges:
            lifecycle = math.ceil(
                (max(edge[2]['timestamp'] for edge in edges) - min(edge[2]['timestamp'] for edge in edges)) / (
                        24 * 60 * 60))
        else:
            lifecycle = 0
        activity_period = []
        last_date = None
        for _, _, data in edges:
            edge_date = datetime.fromtimestamp(data['timestamp']).strftime('%Y-%m-%d')
            if last_date is None or last_date != edge_date:
                activity_period.append(edge_date)
            last_date = edge_date
        active_instances = [0] * len(activity_period)
        for _, _, data in edges:
            edge_date = datetime.fromtimestamp(data['timestamp']).strftime('%Y-%m-%d')
            active_instances[activity_period.index(edge_date)] += 1
        transaction_gaps = []
        for i in range(len(edges) - 1):
            gap = (edges[i + 1][2]['timestamp'] - edges[i][2]['timestamp']) / (24 * 60 * 60)
            transaction_gaps.append(gap)

        return lifecycle, len(activity_period), active_instances, transaction_gaps


def calculate_transaction_metrics(g, node, input_amounts, output_amounts):
    in_edges = sorted(g.in_edges(node, data=True), key=lambda x: x[2].get('timestamp', 0))
    out_edges = sorted(g.out_edges(node, data=True), key=lambda x: x[2].get('timestamp', 0))
    in_daily_edges = defaultdict(list)
    out_daily_edges = defaultdict(list)
    for edge in in_edges:
        date = datetime.fromtimestamp(edge[2].get('timestamp', 0)).strftime('%Y-%m-%d')
        in_daily_edges[date].append(edge)
    for edge in out_edges:
        date = datetime.fromtimestamp(edge[2].get('timestamp', 0)).strftime('%Y-%m-%d')
        out_daily_edges[date].append(edge)
    daily_input_degree = []
    daily_output_degree = []
    daily_input_volumes = []
    daily_output_volumes = []
    for date, edges in in_daily_edges.items():
        input_degree = len(edges)
        input_volume = sum(edge[2]['amount'] for edge in edges)
        daily_input_degree.append(input_degree)
        daily_input_volumes.append(input_volume)

    for date, edges in out_daily_edges.items():
        output_degree = len(edges)
        output_volume = sum(edge[2]['amount'] for edge in edges)
        daily_output_degree.append(output_degree)
        daily_output_volumes.append(output_volume)
    change_in_degree = [daily_input_degree[i] - daily_input_degree[i - 1] if i > 0 else daily_input_degree[i] for i in
                        range(len(daily_input_degree))]
    change_out_degree = [daily_output_degree[i] - daily_output_degree[i - 1] if i > 0 else daily_output_degree[i] for i
                         in range(len(daily_output_degree))]
    in_ratios = calculate_change_and_interval(in_edges)
    out_ratios = calculate_change_and_interval(out_edges)

    in_time_intervals, in_degree_changes = degree_change_and_interval(in_edges)
    out_time_intervals, out_degree_changes = degree_change_and_interval(out_edges)
    in_degree_ratios = [change / interval for change, interval in zip(in_degree_changes, in_time_intervals) if
                        interval > 0]
    out_degree_ratios = [change / interval for change, interval in zip(out_degree_changes, out_time_intervals) if
                         interval > 0]

    in_time_interval = [
        (input_amounts / (change / interval)) if interval > 0 and change != 0 else 0  # 计算 input_amounts 除以 (change /
        # interval)
        for change, interval in zip(in_degree_changes, in_time_intervals)
    ]

    out_time_interval = [
        (input_amounts / change / interval) if interval != 0 and change != 0 else 0
        for change, interval in zip(in_degree_changes, in_time_intervals)
    ]
    return daily_input_degree, daily_output_degree, daily_input_volumes, daily_output_volumes, change_in_degree, change_out_degree, in_ratios, out_ratios, in_degree_ratios, out_degree_ratios, in_time_interval, out_time_interval


def calculate_difference(input_total, output_total):
    difference = input_total - output_total
    return difference
def calculate_ratio(num1, num2):
    if isinstance(num1, list) and isinstance(num2, list):
        if len(num1) != len(num2):
            raise ValueError("When two lists are operated on, they must be the same length")
        ratio = [(n / m) if m != 0 else 0 for n, m in zip(num1, num2)]
    elif isinstance(num1, list):
        ratio = [(n / num2) if num2 != 0 else 0 for n in num1]
    elif isinstance(num2, list):
        ratio = [(num1 / m) if m != 0 else 0 for m in num2]
    else:
        ratio = (num1 / num2) if num1 is not None and num2 is not None and num2 != 0 else 0

    return ratio


def concatenate_lists(list1, list2):
    combined_list = [item for sublist in [list1, list2] for item in sublist]
    return combined_list


def min_and_max(input_amounts):

    if isinstance(input_amounts, (int, float)):
        return input_amounts, input_amounts, input_amounts

    if input_amounts is None:
        return 0, 0, 0

    if not input_amounts:
        return 0, 0, 0
    min_value = min(input_amounts)
    max_value = max(input_amounts)
    avg_value = sum(input_amounts) / len(input_amounts) if input_amounts else 0
    return min_value, max_value, avg_value


def get_node_degrees(graph):
    out_degrees = [graph.out_degree(n) for n in graph.nodes()]
    in_degrees = [graph.in_degree(n) for n in graph.nodes()]
    total_degrees = [out + in_ for out, in_ in zip(out_degrees, in_degrees)]
    return out_degrees, in_degrees, total_degrees


def pearson_correlation_for_node(graph: nx.Graph, node: int):
    if graph.degree(node) == 0:
        return 0
    neighbor_degrees = [graph.degree(neighbor) for neighbor in graph.neighbors(node)]
    if len(neighbor_degrees) == 0:
        return 0
    node_degree = graph.degree(node)
    mean_neighbor_degree = np.mean(neighbor_degrees)
    mean_node_degree = node_degree
    numerator = sum((d - mean_neighbor_degree) * (node_degree - mean_node_degree) for d in neighbor_degrees)
    denominator = np.sqrt(sum((d - mean_neighbor_degree) ** 2 for d in neighbor_degrees))
    if denominator == 0:
        return 0
    return numerator / denominator


def calculate_all_nodes_correlation(graph: nx.Graph):
    correlation_data = {}
    for node in graph.nodes():
        correlation_data[node] = pearson_correlation_for_node(graph, node)
    return correlation_data


def pearson_correlation(g, filename='pearson_correlation.pkl'):
    if not os.path.exists(filename):
        print("Pearson correlation coefficients are calculated for all nodes")
        all_nodes_correlation = calculate_all_nodes_correlation(g)
        save_with_pickle(all_nodes_correlation, filename)
    else:
        all_nodes_correlation = load_with_pickle(filename)
    return all_nodes_correlation


def precompute_and_save_betweenness_centrality(graph: nx.Graph, filename="betweenness_centrality.pkl"):
    """Precompute and save the betweenness centrality for all nodes in the graph using pickle if not exists."""
    if not os.path.exists(filename):
        betweenness_centrality = nx.betweenness_centrality(graph, weight=None)
        save_with_pickle(betweenness_centrality, filename)
        return betweenness_centrality
    else:
        return load_betweenness_centrality(filename)


def save_with_pickle(data, filename):
    """Save data to a file using pickle."""
    with open(filename, 'wb') as f:
        pickle.dump(data, f)


def load_with_pickle(filename):
    """Load data from a file using pickle."""
    with open(filename, 'rb') as f:
        return pickle.load(f)


def load_betweenness_centrality(filename="betweenness_centrality.pkl"):
    """Load precomputed betweenness centrality from a pickle file."""
    betweenness_centrality = load_with_pickle(filename)
    return betweenness_centrality


def get_node_betweenness_centrality(betweenness_centrality, node):
    """Retrieve the betweenness centrality of a specific node from the precomputed dictionary."""
    return betweenness_centrality.get(node, None)


def calculate_stdev(data):
    if isinstance(data, Number):
        return 0
    elif hasattr(data, '__iter__') and not isinstance(data, str):
        try:
            stdev = statistics.stdev(data)
            return stdev
        except statistics.StatisticsError:
            return 0
    else:
        raise TypeError("Data should be a list of numbers or a single number.")


def calculate_change_and_interval(edges):
    changes = []
    for i in range(len(edges) - 1):
        amount_change = edges[i + 1][2]['amount'] - edges[i][2]['amount']
        time_interval = edges[i + 1][2]['timestamp'] - edges[i][2]['timestamp']
        if time_interval > 0:
            changes.append(amount_change / time_interval)
        else:
            changes.append(0)
    return changes


def calculate_degrees(G, node):
    if node not in G:
        return None, None, None


    in_degree = G.in_degree(node)
    out_degree = G.out_degree(node)

    total_degree = in_degree + out_degree

    return in_degree, out_degree, total_degree


def average_path_length_for_component(G):
    if len(G) < 2:
        return 0
    path_lengths = dict(nx.all_pairs_shortest_path_length(G))
    total_length = 0
    count = 0
    for node in G:
        for neighbor in G[node]:
            if neighbor > node:
                total_length += path_lengths[node][neighbor]
                count += 1
    if count == 0:
        return 0
    return total_length / count


def global_average_path_length(G, filename="global_average_path_length.pkl"):
    if os.path.exists(filename):
        return load_with_pickle(filename)
    else:
        total_avg_path_length = 0
        total_nodes = 0

        if nx.is_directed(G):
            strongly_connected_components = list(nx.strongly_connected_components(G))
        else:
            strongly_connected_components = [G]

        for component in strongly_connected_components:
            subgraph = G.subgraph(component)
            avg_path_length = average_path_length_for_component(subgraph)
            num_nodes = len(subgraph)
            total_avg_path_length += avg_path_length * num_nodes
            total_nodes += num_nodes

        if total_nodes == 0:
            average_path_length = 0
        else:
            average_path_length = total_avg_path_length / total_nodes

        save_with_pickle(average_path_length, filename)
        return average_path_length


def module_226(graph: nx.Graph, node_index):

    def local_clustering(g, v):
        neighbors = list(g.neighbors(v))
        triangles = 0
        if len(neighbors) > 1:
            subgraph = g.subgraph([v] + neighbors)
            for n1, n2 in subgraph.edges():
                if g.has_edge(n1, n2):
                    triangles += 1
        if len(neighbors) > 1:
            clustering_coefficient = 2 * triangles / (len(neighbors) * (len(neighbors) - 1))
        else:
            clustering_coefficient = 0

        return clustering_coefficient

    if node_index in graph:
        return local_clustering(graph, node_index)
    else:
        return f"node {node_index} doesn't exist in the graph。"


def precompute_closeness_centrality(graph: nx.Graph, filename="closeness_centrality.pkl"):
    if os.path.exists(filename):
        return load_with_pickle(filename)
    else:
        closeness_centrality = nx.closeness_centrality(graph)
        save_with_pickle(closeness_centrality, filename)
        return closeness_centrality


def get_node_closeness_centrality(closeness_centrality, node):
    return closeness_centrality.get(node)


def module_233(graph: nx.Graph, node: int):
    pr = nx.pagerank(graph)
    node_pagerank = pr.get(node)
    if node_pagerank is not None:
        return node_pagerank
    else:
        print(f"Node {node} not found in the graph.")
        return None


def calculate_directed_graph_density(G):
    if not isinstance(G, nx.DiGraph):
        raise TypeError("The graph must be a directed graph.")

    n = G.number_of_nodes()
    e = G.number_of_edges()
    density = e / (n * (n - 1))

    return density


def degree_change_and_interval(edges):
    time_intervals = []
    degree_changes = []
    for i in range(len(edges) - 1):
        time_interval = edges[i + 1][2]['timestamp'] - edges[i][2]['timestamp']
        degree_change = edges[i + 1][2]['amount'] - edges[i][2]['amount']
        time_intervals.append(time_interval)
        degree_changes.append(degree_change)
    return time_intervals, degree_changes


def find_max_degrees_values_in_component(graph, node):
    components = list(nx.strongly_connected_components(graph))
    component_nodes = [comp for comp in components if node in comp]
    if not component_nodes:
        raise ValueError(f"Node {node} is not in any strongly connected component.")
    subgraph = graph.subgraph(component_nodes[0])
    max_indegree_value = max(subgraph.in_degree(), key=lambda x: x[1])[1]
    max_outdegree_value = max(subgraph.out_degree(), key=lambda x: x[1])[1]
    max_total_degree_value = max([deg for _, deg in subgraph.degree()], key=lambda deg: deg)

    return max_indegree_value, max_outdegree_value, max_total_degree_value
