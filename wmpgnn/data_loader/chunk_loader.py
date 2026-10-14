import gc
import glob
import os
from tqdm import tqdm

from multiprocessing.pool import ThreadPool
from functools import partial
from itertools import chain

import numpy as np
import pytorch_lightning as pl
import torch
from torch.utils.data import IterableDataset
from torch_geometric.loader import DataLoader

from wmpgnn.data_loader.helper import *


class ChunkDataset(IterableDataset):

    def __init__(self, file_paths, configs, mode="train", n_chunks=32):
        super().__init__()
        self.file_paths = file_paths
        self.n_chunks = n_chunks
        self.mode = mode
        self.configs = configs
        cumulative_sizes = [0]
        total = 0
        file_index = []

        for i, key in enumerate(self.file_paths.keys()):
            total += len(self.file_paths[key])
            cumulative_sizes.append(total)
            file_index.append(self._generate_groups(n_chunks=self.n_chunks,
                                                    low=cumulative_sizes[i],
                                                    high=cumulative_sizes[i + 1]))
        self.chunk_index = torch.cat(file_index, dim=1).to(torch.long)
        self.files_per_chunk = self.chunk_index.shape[1]
        self.n_files = {}
        for sample in self.file_paths.keys():
            self.n_files[sample] = len(self.file_paths[sample])
        self.file_paths = list(chain.from_iterable(self.file_paths[sample] for sample in self.file_paths.keys()))

        self.seeds = torch.randint(0, 1000, (1000,))
        self.seed_tracker = 0

    @staticmethod
    def _generate_groups(n_chunks: int, low: int, high: int)-> torch.Tensor:
        groups = []
        available = np.arange(low, high)
        interval_size = high - low
        chunksize = int(np.ceil(interval_size / n_chunks))

        for i in range(n_chunks):
            if len(available) >= chunksize:
                indices = np.random.choice(len(available), size=chunksize, replace=False)
                group = available[indices]
                available = np.delete(available, indices)
            else:
                if len(available) > 0:
                    group = np.concatenate([
                        available,
                        np.random.choice(np.arange(low, high), size=chunksize - len(available), replace=True)
                    ])
                else:
                    group = np.random.choice(np.arange(low, high), size=chunksize, replace=True)
                available = np.array([])
            groups.append(group.tolist())
        return torch.tensor(groups)

    def _load_chunk(self, chunk_number, mode="loading"):
        nevnts = 800
        if self.mode == "validation" or self.mode == "test":
            nevnts = 200

        chunk = self.chunk_index[chunk_number]
        files = [self.file_paths[i] for i in chunk]
        desc = f"Loading {self.mode} chunk {chunk_number + 1}/{self.n_chunks} ({self.files_per_chunk} files, ~{self.files_per_chunk * nevnts} events)"
        if mode == "loading":
            dataset = []
            load_dataset_part = partial(load_dataset, configs=self.configs, mode=mode)
            with ThreadPool(processes=self.configs["settings"]["ncpu"]) as pool:
                for r in tqdm(pool.imap(load_dataset_part, files), total=len(files), desc=desc, leave=False):
                    dataset.extend(r)
            return dataset
        elif mode == "weights":
            weights = {}
            load_dataset_part = partial(load_dataset, configs=self.configs, mode="weights_only")
            with ThreadPool(processes=self.configs["settings"]["ncpu"]) as pool:
                for r in tqdm(pool.imap(load_dataset_part, files), total=len(files), desc=desc, leave=False):
                    for key, value in r.items():
                        if key not in weights:
                            weights[key] = value
                        else:
                            weights[key] += value
            return weights
        else:
            raise NotImplementedError

    def __iter__(self):

        worker_info = torch.utils.data.get_worker_info()
        if worker_info is None:
            iter_start = 0
            iter_end = self.n_chunks
        else:
            worker_id = worker_info.id
            per_worker = int(np.ceil(self.n_chunks / worker_info.num_workers))
            iter_start = worker_id * per_worker
            iter_end = min(iter_start + per_worker, self.n_chunks)

        for chunk_number in range(iter_start, iter_end):

            chunk_events = self._load_chunk(chunk_number)


            idx = torch.arange(0, len(chunk_events))
            if self.mode == "train":
                idx = idx[torch.randperm(len(idx))]
            else:

                g = torch.Generator()
                g.manual_seed(self.seeds[self.seed_tracker].item())
                self.seed_tracker += 1
                idx = idx[torch.randperm(len(idx), generator=g)]


            for i, i_idx in enumerate(idx):
                yield chunk_events[i_idx]

            del chunk_events
            gc.collect()

    def get_weights(self):
        weights = {}
        for i in range(self.n_chunks):
            weights[i] = self._load_chunk(i, mode="weights")
        return weights


class ChunkLoader(pl.LightningDataModule):

    def __init__(self, configs, trn_dataset=None, val_dataset=None, tst_dataset=None, batch_size=None, num_workers=None):
        super().__init__()
        self.trn_dataset = trn_dataset
        self.val_dataset = val_dataset
        self.tst_dataset = tst_dataset
        self.configs = configs

        if isinstance(batch_size, int):
            self.batch_size = batch_size
        else:
            self.batch_size = self.configs["settings"]["batch_size"]
        if isinstance(num_workers, int):
            self.num_workers = num_workers
        else:
            self.num_workers = self.configs["settings"]["ncpu"] * 2

    def train_dataloader(self):
        if not isinstance(self.trn_dataset, type(None)):
            trn_loader = DataLoader(self.trn_dataset, batch_size=self.batch_size, num_workers=self.num_workers,
                                    drop_last=True, persistent_workers=False,
                                    pin_memory=False)
        else:
            raise ValueError("trn_dataset must not be None")
        return trn_loader

    def val_dataloader(self):
        if not isinstance(self.val_dataset, type(None)):
            val_loader = DataLoader(self.val_dataset, batch_size=self.batch_size, num_workers=self.num_workers,
                                    drop_last=True, persistent_workers=False,
                                    pin_memory=False)
        else:
            raise ValueError("val_dataset must not be None")
        return val_loader

    def test_dataloader(self):
        if not isinstance(self.tst_dataset, type(None)):
            tst_loader = DataLoader(self.tst_dataset, batch_size=self.batch_size, num_workers=self.num_workers,
                                    drop_last=False, persistent_workers=False,
                                    pin_memory=False)
        else:
            raise ValueError("tst_dataset not must be None")
        return tst_loader


def get_trn_val_loaders(_configs) -> ChunkLoader:
    data_dir = _configs["settings"]["data_dir"]
    num_workers = _configs["settings"]["ncpu"] * 2
    nfiles = get_nfiles(_configs["settings"])

    """Training"""
    path_dict = {}
    for sample, files in nfiles.items():
        path_dict[sample] = sorted(glob.glob(f'{data_dir}/{sample}/trn_data_*'))[:files]


    min_files\
        = min(len(v) for v in path_dict.values() if len(v) > 0)
    total_files = sum(len(v) for v in path_dict.values())
    num_chunks = np.ceil(min_files / num_workers).astype(int)

    while total_files / num_chunks >= 8:
        num_chunks += num_workers
    print(f"Number of chunks: {num_chunks}")
    print(f"Files per chunk: {np.ceil(total_files / num_chunks).astype(int)}")

    trn_dataset = ChunkDataset(path_dict, _configs, mode="train", n_chunks=num_chunks)

    """Validation"""
    path_dict = {}
    for sample, files  in nfiles.items():
        path_dict[sample] = sorted(glob.glob(f'{data_dir}/{sample}/val_data_*'))[:5]
    val_dataset = ChunkDataset(path_dict, _configs, mode="validation", n_chunks=num_chunks)

    return ChunkLoader(_configs, trn_dataset=trn_dataset, val_dataset=val_dataset)


def get_tst_loader(_configs) -> ChunkLoader:
    data_dir = _configs["settings"]["data_dir"]
    nfiles = get_nfiles(_configs["evaluate"])
    split = _configs["evaluate"].get("split", "tst")
    if split not in {"val", "tst"}:
        raise ValueError(f"Unknown evaluation split: {split}")

    path_dict = {"testing": []}
    for sample, files in nfiles.items():
        path_dict["testing"].append(sorted(glob.glob(f'{data_dir}/{sample}/{split}_data_*'))[:files])
    path_dict["testing"] = sum(path_dict["testing"], [])

    # Eval batch. 512 spikes RAM/GPU on big-graph datasets (Bs-200k SIGKILL/OOM);
    # 128 is safe and eval is not throughput-critical. Override via EVAL_BATCH.
    batch_size = int(os.environ.get("EVAL_BATCH", "128"))
    tst_dataset = ChunkDataset(path_dict, _configs, mode="test", n_chunks=len(path_dict["testing"]))

    return ChunkLoader(_configs, tst_dataset=tst_dataset, batch_size=batch_size, num_workers=2)
