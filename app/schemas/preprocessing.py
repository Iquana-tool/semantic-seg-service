from typing import Optional

import numpy as np
import torch
from pydantic import BaseModel, Field


class Preprocessing(BaseModel):
    """ Preprocessing is anything that should be done to the uploaded images exactly once. These are usually performance
        heavy computations like Histogram equalization. This is a placeholder for now."""

    def __call__(self, img: np.ndarray) -> np.ndarray:
        return img
