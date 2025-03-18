# -*- coding:utf-8 -*-
import numpy as np
from gensim.models import Word2Vec
from gensim.models.callbacks import CallbackAny2Vec

from ..walker import RandomWalker  


class EpochLogger(CallbackAny2Vec):
    def __init__(self):
        self.epoch = 0

    def on_epoch_end(self, model):
        print(f"Epoch #{self.epoch} end")
        self.epoch += 1


class Node2Vec:

    def __init__(self, graph, transaction_volume, transaction_volume_between, walk_length, num_walks, workers):
        if not graph:
            raise ValueError("Graph cannot be None")
        if walk_length <= 0:
            raise ValueError("walk_length must be positive")
        if num_walks <= 0:
            raise ValueError("num_walks must be positive")
        if workers <= 0:
            raise ValueError("workers must be positive")

        self.graph = graph
        self._embeddings = {}
        self.transaction_volume = transaction_volume
        self.transaction_volume_between = transaction_volume_between
        self.walker = RandomWalker(graph, transaction_volume, transaction_volume_between)

        print("Preprocess transition probs...")
        
        self.walker.preprocess_transition_probs()

        self.sentences = self.walker.simulate_walks(
            num_walks=num_walks, walk_length=walk_length, workers=workers, verbose=1)

    def train(self, embed_size=128, window_size=10, workers=10, iter=5, verbose=1, **kwargs):
        kwargs["sentences"] = self.sentences
        kwargs["min_count"] = kwargs.get("min_count", 0)
        kwargs["vector_size"] = embed_size
        kwargs["sg"] = 1
        kwargs["hs"] = 0  # node2vec not use Hierarchical Softmax
        kwargs["workers"] = workers
        kwargs["window"] = window_size
        kwargs["epochs"] = iter
        if verbose:
            kwargs["callbacks"] = [EpochLogger()]

        print("Learning embedding vectors...")
        model = Word2Vec(**kwargs)
        print("Learning embedding vectors done!")

        self.w2v_model = model

        return model

    def get_embeddings(self):
        if not hasattr(self, 'w2v_model') or not hasattr(self.w2v_model, 'wv'):
            print("Model not trained or does not have 'wv' attribute")
            return {}

        self._embeddings = {}
        for node in self.graph.nodes():
            
            if not isinstance(node, (int, str)):
                print(f"Node '{node}' is not a valid type for a dictionary key.")
                continue

           
            try:
                vector = self.w2v_model.wv[node]
            except KeyError:
                print(f"Word '{node}' not found in the model's vocabulary.")
                continue
            except TypeError as e:
                print(f"TypeError encountered for node '{node}': {e}")
                continue

            
            if not isinstance(vector, (list, np.ndarray)):
                print(f"Vector for node '{node}' is not a list or NumPy array: {type(vector)}")
                continue

            self._embeddings[node] = vector

        return self._embeddings
