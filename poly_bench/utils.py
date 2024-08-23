from typing import Optional
from jaxtyping import Float, Int

from datasets import Dataset # type: ignore
import torch as t
import numpy as np

class SimpleDataset(Dataset):
    def __init__(self, 
                 inputs: Int[np.ndarray | t.Tensor, "batch seq"], 
                 targets: Float[np.ndarray | t.Tensor, "batch seq d_vocab"],
                 markers: Optional[Int[np.ndarray | t.Tensor, "batch seq"]] = None
                 ):
        self.inputs = t.tensor(inputs).to(t.int)
        self.targets = t.tensor(targets).to(t.float32)
        if markers is None:
            self.markers = None
        else:
            self.markers = t.tensor(markers).to(t.int)

    def __len__(self) -> int:
        return len(self.inputs)

    def __getitem__(self, idx: int) -> tuple[Int[t.Tensor, "batch seq"], Float[t.Tensor, "batch seq d_vocab"], Optional[Int[t.Tensor, "batch seq"]]]:
        if self.markers is None:
            return self.inputs[idx], self.targets[idx], None
        else:
            return self.inputs[idx], self.targets[idx], self.markers[idx]

