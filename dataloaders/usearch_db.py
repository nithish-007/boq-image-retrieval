from usearch.index import Index
import numpy as np

class UsearchDB:
    def __init__(self, embedding_dim, metric, dtype):
        self.index = Index(ndim=embedding_dim, metric=metric, dtype=dtype)
        self.labels = []
        self.next_key = 0  # Track unique key counter

    def add(self, embeddings, labels):
        """Add batch of embeddings with unique keys"""
        # Generate unique keys for this batch
        num_embeddings = len(embeddings)
        keys = np.arange(self.next_key, self.next_key + num_embeddings)
        
        # Batch add to index
        self.index.add(keys, embeddings)
        
        # Store labels and update key counter
        self.labels.extend(labels.tolist())
        self.next_key += num_embeddings

    def search(self, query_vector, k=5):
        """Search with proper numpy conversion"""
        return self.index.search(query_vector, k)