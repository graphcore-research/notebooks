# Copyright (c) 2022 Graphcore Ltd. All rights reserved.

"""Custom op driver code, and demo application, for static sparse @ dense matrix multiplication on IPU."""

import ctypes
from pathlib import Path
import poptorch
import torch
import numpy as np

NUMPY_TO_PYTORCH_DTYPE = {np.dtype("float32"): torch.float, np.dtype("float16"): torch.half}

def dynamic_spmm_ipu(sparse: torch.Tensor, dense: torch.Tensor) -> torch.Tensor:
    row_indices, col_indices = sparse.indices().numpy()
    nzvalues = sparse.values().numpy().flatten()
    n_rows = sparse.shape[0] * sparse.shape[2]
    n_cols = sparse.shape[1] * sparse.shape[3]
    block_size = sparse.shape[2]
    if dense.shape[0] != n_cols:
        raise ValueError(
            f"Bad StaticSparseTensor multiply, ({n_rows}, {n_cols}) @ {tuple(dense.shape)}"
        )
    batch_size = dense.shape[1]
    library_path = (
        Path(__file__).parent.absolute() / "libpoptorch_dynamic_sparse_op.so"
    )
    ctypes.cdll.LoadLibrary(library_path)

    (y,) = poptorch.custom_op(
        [dense],
        "StaticDynSparse",
        "ai.graphcore",
        1,
        example_outputs=[
            torch.zeros((n_rows, batch_size), dtype=dense.dtype, device=dense.device)
        ],
        attributes=dict(
            n_rows=n_rows,
            n_cols=n_cols,
            block_size=block_size,
            rows=row_indices.tolist(),
            cols=col_indices.tolist(),
            values=nzvalues.tolist(),
        ),
    )

    return y




