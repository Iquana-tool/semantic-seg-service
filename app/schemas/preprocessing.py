from typing import Optional

import torch
from pydantic import BaseModel, Field


class Preprocessing(BaseModel):
    normalize_mean_and_std: Optional[tuple[tuple[float, float, float], tuple[float, float, float]]] = Field(
        default=None, description="Normalize by this mean and standard deviation. If None, does not normalize.")

    def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
        if self.normalize:
            tensor = torch.transforms.normalize(tensor, self.mean, self.std)
        return tensor
