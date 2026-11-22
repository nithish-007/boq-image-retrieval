import os
import cv2
import numpy as np 
from scipy.spatial.distance import cosine, cdist
from .boq_dataset import TrainCustomDataset
from collections import defaultdict

class HistogramCentroid:
    def __init__(self, dataset: TrainCustomDataset):
        self.dataset = dataset
        self.color_spectrum = np.array([
            (0, 255, 255), (15, 255, 255), (30, 255, 255), (45, 255, 255), (60, 255, 255),
            (75, 255, 255), (90, 255, 255), (105, 255, 255), (120, 255, 255), (135, 255, 255),
            (150, 255, 255), (165, 255, 255), (180, 255, 255), (195, 255, 255), (210, 255, 255),
            (225, 255, 255), (240, 255, 255), (255, 255, 255), (270, 255, 255), (285, 255, 255),
            (300, 255, 255), (315, 255, 255), (330, 255, 255), (345, 255, 255), (360, 255, 255),
            (0, 200, 200), (60, 200, 200), (120, 200, 200), (180, 200, 200), (240, 200, 200),
            (300, 200, 200), (0, 150, 150), (60, 150, 150), (120, 150, 150), (180, 150, 150),
            (240, 150, 150), (300, 150, 150), (0, 100, 100), (60, 100, 100), (120, 100, 100),
            (180, 100, 100), (240, 100, 100), (300, 100, 100), (0, 50, 50), (60, 50, 50),
            (120, 50, 50), (180, 50, 50), (240, 50, 50), (300, 50, 50)
        ])
        self.centroids = self.compute_class_centroids()
        self._precompute_similarity_matrix()

    def compute_histogram(self, pil_image):
        """Compute color histogram from PIL Image"""
        # Convert to OpenCV format
        img_np = np.array(pil_image)
        img_np = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
        hsv_image = cv2.cvtColor(img_np, cv2.COLOR_BGR2HSV)
        
        # Compute histogram
        reshaped_image = hsv_image.reshape(-1, 3)
        distances = cdist(reshaped_image, self.color_spectrum, metric="euclidean")
        closest_indices = np.argmin(distances, axis=1)
        distribution = np.bincount(closest_indices, minlength=len(self.color_spectrum))
        return distribution / len(reshaped_image)

    def compute_class_centroids(self):
        """Compute centroids using only original images"""
        class_histograms = defaultdict(list)
        
        # Only use original images for centroid calculation
        for img, label in zip(self.dataset.original_images, self.dataset.original_labels):
            hist = self.compute_histogram(img)
            class_histograms[label].append(hist)
            
        centroids = {}
        for label, hists in class_histograms.items():
            centroids[label] = np.mean(hists, axis=0)
        return centroids
    
    def _precompute_similarity_matrix(self):
        labels = sorted(self.centroids.keys())
        centroid_vecs = np.array([self.centroids[l] for l in labels])
        
        # Normalize vectors
        norms = np.linalg.norm(centroid_vecs, axis=1, keepdims=True) + 1e-8
        norm_centroids = centroid_vecs / norms
        
        # Compute cosine similarity matrix
        self.sim_matrix = norm_centroids @ norm_centroids.T
        self.label_order = labels
        self.label_to_idx = {l: i for i, l in enumerate(labels)}
        
        # Precompute sorted neighbors
        self.neighbor_cache = {}
        for i, label in enumerate(labels):
            similarities = self.sim_matrix[i]
            sorted_indices = np.argsort(similarities)[::-1]  # Descending order
            self.neighbor_cache[label] = [
                labels[j] for j in sorted_indices 
                if j != i  # Exclude self
            ]

    def find_nearest_classes(self, target_label, N):
        return self.neighbor_cache.get(target_label, [])[:N]
