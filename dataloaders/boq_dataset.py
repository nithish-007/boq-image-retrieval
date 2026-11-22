import os 
from collections import defaultdict
from typing import Tuple
import random 

import cv2
import numpy as np
from PIL import Image

import torch 
from torch.utils.data import Dataset
from torchvision.datasets import ImageFolder
from torchvision.transforms import transforms as T
from PIL import UnidentifiedImageError

from torch.utils.data import random_split
from .boq_transform import augmentation, get_boq_transform
    
# Training Dataset
class TrainCustomDataset(Dataset):
    def __init__(self, input_dir:str, train_augmentation:dict):
        self.images = []
        self.labels = []
        self.image_info = []
        self.label_to_indices = defaultdict(list)
        self.idx_to_class = {}
        self.class_to_idx = {}

        # Define augmentation pipeline
        self.transform = augmentation(train_augmentation)

        # Collect class names
        self.class_names = sorted(
            [d for d in os.listdir(input_dir) 
             if os.path.isdir(os.path.join(input_dir, d))]
        )
        self.class_to_idx = {cls_name: idx for idx, cls_name in enumerate(self.class_names)}
        self.idx_to_class = {idx: cls_name for idx, cls_name in enumerate(self.class_names)}

        # Track original images for centroid calculation
        self.original_images = []
        self.original_labels = []
        
        # Process each class
        for cls_name in self.class_names:
            cls_dir = os.path.join(input_dir, cls_name)
            cls_idx = self.class_to_idx[cls_name]
            
            # Process original images
            orig_paths = [
                os.path.join(cls_dir, fname) 
                for fname in os.listdir(cls_dir) 
                if fname.lower().endswith(('.jpg', '.jpeg', '.png'))
            ]
            
            for path in orig_paths:
                img = Image.open(path).convert('RGB')
                idx = len(self.images)
                
                # Store in main dataset
                self.images.append(img)
                self.labels.append(cls_idx)
                self.image_info.append((cls_idx, False, idx))
                self.label_to_indices[cls_idx].append(idx)
                
                # Store separately for centroid calculation
                self.original_images.append(img)
                self.original_labels.append(cls_idx)
            
            # Augment if needed
            if len(orig_paths) <= 2:
                num_needed = 8 - len(orig_paths)
                augmented_images = self.augment_class_images(cls_idx, num_needed)
                
                for aug_img in augmented_images:
                    idx = len(self.images)
                    self.images.append(aug_img)
                    self.labels.append(cls_idx)
                    self.image_info.append((cls_idx, True, idx))
                    self.label_to_indices[cls_idx].append(idx)

    def augment_class_images(self, class_idx, num_needed):
        """Generate augmented images for a class in memory"""
        augmented = []
        class_indices = [i for i, (cid, _, _) in enumerate(self.image_info) 
                         if cid == class_idx and not self.image_info[i][1]]
        
        if not class_indices:
            return augmented
        
        # Use existing class images for augmentation
        for _ in range(num_needed):
            orig_idx = random.choice(class_indices)
            img = self.images[orig_idx]
            
            # Apply augmentations
            img_aug = self.transform(img)
            augmented.append(img_aug)
            
        return augmented

    def __getitem__(self, idx):
        # Return PIL Image and label
        return self.images[idx], self.labels[idx]

    def __len__(self):
        return len(self.images)

 
class TripletWithNegativesDataset(Dataset):
    def __init__(self, sampler, transform_config):
        self.samples = list(sampler)
        self.transform = get_boq_transform(transform_config)
        self.dataset = sampler.dataset  # Access to path_to_label

    def __getitem__(self, idx):
        anchor_idx, pos_idx, neg_indices = self.samples[idx]

        # Load and transform images
        anchor_img, anchor_label = self.dataset[anchor_idx]
        pos_img, pos_label = self.dataset[pos_idx]
        
        anchor_tensor = self.transform(anchor_img)
        pos_tensor = self.transform(pos_img)
        neg_tensors = [self.transform(self.dataset[i][0]) for i in neg_indices]
        neg_labels = [self.dataset[i][1] for i in neg_indices]
        
        return (
            anchor_tensor, 
            pos_tensor, 
            torch.stack(neg_tensors),
            torch.tensor(anchor_label),
            torch.tensor(pos_label),
            torch.tensor(neg_labels)
        )

    def __len__(self):
        return len(self.samples)
    
# Validattion Dataset    
class RetrievalValidationDataset(Dataset):
    def __init__(self, candidate_dir, query_dir, transform_config):
        self.transform = get_boq_transform(transform_config)
        self.candidate_data = self._load_candidates(candidate_dir)
        self.query_samples = self._load_queries(query_dir)

    def _load_candidates(self, root_dir):
        """Load all candidate images grouped by UPC"""
        data = defaultdict(list)
        for upc in os.listdir(root_dir):
            upc_path = os.path.join(root_dir, upc)
            if os.path.isdir(upc_path):
                for img_file in os.listdir(upc_path):
                    if img_file.lower().endswith(('.jpg', '.jpeg', '.png')):
                        data[upc].append(os.path.join(upc_path, img_file))
        return data

    def _load_queries(self, root_dir):
        """Load all query images as individual samples"""
        samples = []
        for upc in os.listdir(root_dir):
            upc_path = os.path.join(root_dir, upc)
            if os.path.isdir(upc_path):
                for img_file in os.listdir(upc_path):
                    if img_file.lower().endswith(('.jpg', '.jpeg', '.png')):
                        samples.append((upc, os.path.join(upc_path, img_file)))
        return samples

    def __len__(self):
        return len(self.query_samples)

    def __getitem__(self, idx):
        # Get query sample
        query_upc, query_path = self.query_samples[idx]
        
        # Process query image
        query_img = self.transform(Image.open(query_path).convert("RGB"))
        query_label = torch.tensor(int(query_upc), dtype=torch.long)
        
        # Get candidate images from the same UPC
        same_upc_images = self.candidate_data.get(query_upc, [])
        if not same_upc_images:
            raise RuntimeError(f"No candidates found for query UPC: {query_upc}")
        
        # Randomly select one candidate image
        candidate_path = random.choice(same_upc_images)
        candidate_img = self.transform(Image.open(candidate_path).convert("RGB"))
        candidate_label = torch.tensor(int(query_upc), dtype=torch.long)
        
        return candidate_img, candidate_label, query_img, query_label