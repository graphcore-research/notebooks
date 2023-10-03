import dataclasses
import math
import os
import time
from pathlib import Path
from typing import Any, Dict, Union

import numpy as np
import torch
import wandb


class WandBLogger:
    """Weights & Biases logging"""

    def __init__(self, mod, dlrm_config, exec_config, train_config=None) -> None:
        self.args = dict(**dataclasses.asdict(dlrm_config))
        self.args.update(dataclasses.asdict(exec_config))
        if train_config:
            self.args.update(dataclasses.asdict(train_config))
        self.step = 0
        self.sample = 0
        os.environ["WANDB_SILENT"] = "true"
        os.environ["WANDB_DIR"] = "/tmp/wandb"
        Path("./tmp/wandb").mkdir(parents=True, exist_ok=True)
        wandb.init(
            project="compressed_DLRM",  # project name
            config=self.args,
            reinit=True,
        )
        wandb.summary["n_emb_parameters"] = mod.n_emb_parameters

    def log(self, event: str, data: Dict[str, Any]) -> None:
        data = data.copy()
        if event == "train_step":
            self.sample = data["train"].pop("samples")

        wandb.log(
            data,
            step=self.sample,
        )


# Data pipeline
# Adapted from
# https://github.com/facebookresearch/dlrm/blob/main/dlrm_data_pytorch.py
# https://github.com/facebookresearch/dlrm/blob/main/data_loader_terabyte.py


def _transform_features(
    x_int_batch, x_cat_batch, y_batch, max_ind_range, flag_input_torch_tensor=False
):
    """Bach data preprocessing"""

    # if tables are vertically scaled, categorical value IDs are taken modulo max_ind_range
    if max_ind_range > 0:
        x_cat_batch = x_cat_batch % max_ind_range

    if flag_input_torch_tensor:
        x_int_batch = torch.log(x_int_batch.clone().detach().type(torch.float) + 1)
        x_cat_batch = x_cat_batch.clone().detach().type(torch.long)
        y_batch = y_batch.clone().detach().type(torch.float32).view(-1, 1)
    else:
        x_int_batch = torch.log(torch.tensor(x_int_batch, dtype=torch.float) + 1)
        x_cat_batch = torch.tensor(x_cat_batch, dtype=torch.long)
        y_batch = torch.tensor(y_batch, dtype=torch.float32).view(-1, 1)

    return x_int_batch, x_cat_batch, y_batch.view(-1, 1)


class CriteoBinDataset(torch.utils.data.Dataset):
    """Binary version of Criteo 1TB dataset."""

    def __init__(
        self,
        data_file,
        counts_file,
        batch_size=1,
        max_ind_range=-1,
        bytes_per_feature=4,
    ):
        # Criteo 1TB dataset
        self.tar_fea = 1  # single target
        self.den_fea = 13  # 13 dense  features
        self.spa_fea = 26  # 26 sparse features
        self.tad_fea = self.tar_fea + self.den_fea
        self.tot_fea = self.tad_fea + self.spa_fea

        self.batch_size = batch_size
        self.max_ind_range = max_ind_range
        self.bytes_per_entry = bytes_per_feature * self.tot_fea * batch_size

        self.num_entries = math.floor(os.path.getsize(data_file) / self.bytes_per_entry)

        print("data file:", data_file, "number of batches:", self.num_entries)
        self.file = open(data_file, "rb")

        with np.load(counts_file) as data:
            self.counts = data["counts"]

    def __len__(self):
        return self.num_entries

    def __getitem__(self, idx):
        self.file.seek(idx * self.bytes_per_entry, 0)
        raw_data = self.file.read(self.bytes_per_entry)
        array = np.frombuffer(raw_data, dtype=np.int32)
        tensor = torch.from_numpy(array.copy()).view((-1, self.tot_fea))

        return _transform_features(
            x_int_batch=tensor[:, 1:14],
            x_cat_batch=tensor[:, 14:],
            y_batch=tensor[:, 0],
            max_ind_range=self.max_ind_range,
            flag_input_torch_tensor=True,
        )

    def __del__(self):
        self.file.close()
