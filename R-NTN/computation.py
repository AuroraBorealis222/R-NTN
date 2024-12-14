
import pandas as pd

from Feature_computation import *


def computing(g, node):
    input_amounts, output_amounts = get_node_transactions(g, node)

    Feature1 = sum(input_amounts)
    Feature2 = sum(output_amounts)
    Feature3, Feature4, Feature5 = calculate_degrees(g, node)
    Feature6, Feature7, number, time_interval = calculate_metrics(
        g, node)
    Frequency_transaction = calculate_ratio(Feature5, Feature6)
    Feature8 = calculate_stdev(time_interval)
    Feature9 = calculate_ratio(Feature8, Feature5)
    Feature10 = calculate_ratio(Feature1 + Feature2, Feature5)
    print('node', node)

    results = pd.DataFrame({
        'Node': [node],
        'Feature1': [Feature1],
        'Feature2': [Feature2],
        'Feature3': [Feature3],
        'Feature4': [Feature4],
        'Feature5': [Feature5],
        'Feature8': [Feature8],
        'Frequency_transaction': [Frequency_transaction],
        'Feature9': [Feature9],
        'Feature10': [Feature10],
    })
    return results
