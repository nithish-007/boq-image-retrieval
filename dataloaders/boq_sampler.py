from torch.utils.data import Sampler
from .boq_dataset import TrainCustomDataset
from .hist_centroid import HistogramCentroid
import random 

class TripletWithNegativesSampler(Sampler):
    def __init__(self, dataset: TrainCustomDataset, hist_centroid: HistogramCentroid, sampler_config):
        self.dataset = dataset
        self.hist_centroid = hist_centroid

        self.K = sampler_config["k"]
        self.N = sampler_config["n"]
        self.samples = self.create_samples()

    def create_samples(self):
        samples = []
        anchor_used_indices = set()
        pos_used_indices = set()

        for anchor_idx in range(len(self.dataset)):
            if anchor_idx in anchor_used_indices:
                continue
                
            _, anchor_label = self.dataset[anchor_idx]
            anchor_used_indices.add(anchor_idx)

            # Find positive sample (same class)
            pos_candidates = [
                idx for idx in self.dataset.label_to_indices[anchor_label]
                if idx != anchor_idx and idx not in pos_used_indices
            ]
            
            if not pos_candidates:
                continue
                
            pos_idx = random.choice(pos_candidates)
            pos_used_indices.add(pos_idx)

            # Find negative classes
            nearest_classes = self.hist_centroid.find_nearest_classes(anchor_label, self.N)
            negative_classes = random.sample(nearest_classes, min(self.K, len(nearest_classes)))
            
            neg_indices = []
            for neg_cls in negative_classes:
                # Find available negative samples
                candidates = [
                    idx for idx in self.dataset.label_to_indices[neg_cls]
                    # if idx not in used_indices
                ]
                
                if candidates:
                    neg_idx = random.choice(candidates)
                    neg_indices.append(neg_idx)
                    # used_indices.add(neg_idx)

            samples.append((anchor_idx, pos_idx, neg_indices))

        return samples
    
    def __iter__(self):
        random.shuffle(self.samples)
        for triplet in self.samples:
            yield triplet

    def __len__(self):
        return len(self.samples)
