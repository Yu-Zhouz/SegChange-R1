from .transforms import *
from .loading_data import *
from .building import *
import torch
from utils import collate_fn_building
from torch.utils.data import DataLoader
import logging

def build_dataset(cfg):
    train_set, val_set = loading_data(cfg)

    sampler_train = torch.utils.data.RandomSampler(train_set)  # Random sampling
    sampler_val = torch.utils.data.SequentialSampler(val_set)  # Sequential sampling
    batch_sampler_train = torch.utils.data.BatchSampler(
        sampler_train, cfg.training.batch_size, drop_last=True)
    # DataLoader for training
    data_loader_train = DataLoader(train_set, batch_sampler=batch_sampler_train,
                                   collate_fn=collate_fn_building, num_workers=cfg.num_workers)
    data_loader_val = DataLoader(val_set, 1, sampler=sampler_val,
                                 collate_fn=collate_fn_building, num_workers=cfg.num_workers)
    # Log dataset scanning results
    logging.info("------------------------ preprocess dataset ------------------------")
    logging.info("Data_path: %s", cfg.data.data_root)
    logging.info("Data Transforms:\n %s", cfg.data.transforms)
    logging.info(f"# Train {train_set.nSamples}, Val {val_set.nSamples}")
    return data_loader_train, data_loader_val
