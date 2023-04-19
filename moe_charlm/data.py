# Copyright (c) 2023 Graphcore Ltd. All rights reserved.

from pathlib import Path
import json
from typing import Iterable, Tuple, Optional, Dict

import numpy as np
import torch
import itertools as it


class CharVocab:
    def __init__(self, vocab: Tuple[str, ...]):
        self.vocab = vocab
        self.char_to_idx = {char: id for id, char in enumerate(self.vocab)}

    @classmethod
    def from_path(cls, vocab_path: str) -> "CharVocab":
        """Load a JSON vocabulary file from vocab_path"""
        return cls(tuple(json.loads((Path(vocab_path)).read_text())))

    def str_to_ids(self, text: str) -> torch.Tensor:
        return torch.tensor([self.char_to_idx[c] for c in text], dtype=torch.int64)

    def ids_to_str(self, ids: torch.Tensor) -> str:
        return "".join([self.vocab[i] for i in ids])

    def __len__(self) -> int:
        return len(self.vocab)


class Dataset:
    """Returns batches of shape (batch_size, sequence_length)"""

    def __init__(
        self,
        vocab: CharVocab,
        text: str,
    ):
        self.vocab = vocab
        self.data = self.vocab.str_to_ids(text)

    @classmethod
    def from_path(cls, vocab_path, text_path) -> "Dataset":
        vocab = CharVocab.from_path(vocab_path)
        text = Path(text_path).read_text()
        return Dataset(vocab, text)

    def batch(
        self,
        batch_size: int,
        sequence_length: int,
        overlap_length: int,
        seed: Optional[int] = None,
    ) -> Iterable[Dict[str, torch.Tensor]]:
        shift = sequence_length - overlap_length

        if seed is None:
            starts = range(0, len(self.data) - sequence_length, shift)
        else:
            rnd = np.random.Generator(np.random.PCG64(seed))
            starts = (
                rnd.integers(len(self.data) - sequence_length) for _ in it.count()
            )
        x_batch = []
        y_batch = []
        mask_batch = []
        mask = (torch.arange(sequence_length) >= overlap_length).float()
        for start in starts:
            if len(x_batch) == batch_size:
                yield dict(
                    x=torch.stack(x_batch),
                    y=torch.stack(y_batch),
                    mask=torch.stack(mask_batch),
                )
                x_batch.clear()
                y_batch.clear()
                mask_batch.clear()
            x = self.data[start : start + sequence_length]
            y = self.data[start + 1 : start + sequence_length + 1]
            x_batch.append(x)
            y_batch.append(y)
            mask_batch.append(mask)

    def __len__(self) -> int:
        return len(self.data)
