import os
from collections import defaultdict
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import transforms as T
from torchvision.datasets import ImageFolder

from .boq_dataset import TrainCustomDataset
from .hist_centroid import HistogramCentroid
from .boq_transform import get_boq_transform
from .boq_sampler import TripletWithNegativesSampler
from .boq_dataset import TripletWithNegativesDataset
from .boq_dataset import RetrievalValidationDataset


def get_train_dataloader(dataloader_config):

    dataset = TrainCustomDataset(input_dir=dataloader_config["dataset"]["input_dir"],
                            train_augmentation=dataloader_config["train_augmentation"])
    hist_centroid = HistogramCentroid(dataset=dataset)
    
    # Create samplers for train and val
    train_sampler = TripletWithNegativesSampler(dataset=dataset, hist_centroid=hist_centroid, 
                                                sampler_config=dataloader_config["sampler"])
    
    # Create datasets
    train_triplet_dataset = TripletWithNegativesDataset(sampler=train_sampler, 
                                                        transform_config=dataloader_config["transform"])
    
    # Create dataloaders
    loader_cfg = dataloader_config["dataloader"]
    return DataLoader(
        train_triplet_dataset, 
        batch_size=loader_cfg.get("batch_size", 32),
        shuffle=loader_cfg.get("shuffle", False),
        num_workers=loader_cfg.get("num_workers", 4),
        pin_memory=loader_cfg.get("pin_memory", True),
    )

def get_val_dataloader(dataloader_config):

    # Assuming `dataloader_config` is properly defined
    val_dataset = RetrievalValidationDataset(dataloader_config["dataset"]["candidate_dir"],
                                            dataloader_config["dataset"]["query_dir"],
                                            transform_config = dataloader_config["transform"])
     
    loader_cfg = dataloader_config["dataloader"]
    
    return DataLoader(val_dataset, 
                                batch_size=loader_cfg.get("batch_size", 32), 
                                shuffle=loader_cfg.get("shuffle", False), 
                                num_workers=loader_cfg.get("num_workers", 4), 
                                pin_memory=loader_cfg.get("pin_memory", True))


