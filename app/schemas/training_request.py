import os
from typing import Literal

from pydantic import BaseModel, Field, field_validator

from app.schemas.augmentations import Augmentations
from app.schemas.data_profile import DataProfile
from paths import DATA_PATH
from typing import Literal, Optional, Union
from pydantic import BaseModel, Field
from torch.optim import Adam, SGD, Optimizer
from torch.optim.lr_scheduler import (
    StepLR, ExponentialLR, ReduceLROnPlateau, CosineAnnealingLR, LRScheduler
)


class HyperParams(BaseModel):
    batch_size: int = Field(default=32, description="Batch size to use for training.")
    learning_rate: float = Field(default=0.001, description="Learning rate to use for training.")
    lr_scheduler: Optional[str] = Field(
        default=None,
        description="Learning rate scheduler to use for training. Options: 'step', 'exp', 'plateau', 'cosine'."
    )
    lr_scheduler_step_size: int = Field(
        default=30,
        description="Step size for StepLR scheduler (in epochs)."
    )
    lr_scheduler_gamma: float = Field(
        default=0.1,
        description="Multiplicative factor for StepLR and ExponentialLR schedulers."
    )
    early_stopping_patience: int = Field(
        default=25,
        description="Number of epochs to wait for improvement. If this number is exceeded without the model improving, "
                    "training is stopped regardless of the amount of remaining epochs. If this value is <= 0, "
                    "no early stopping will be applied."
    )
    weight_decay: float = Field(
        default=0.0,
        description="Weight decay (L2 penalty) for optimizer."
    )

    def get_optimizer(self, model_parameters) -> Optimizer:
        """
        Returns a PyTorch optimizer instance based on the hyperparameters.

        Args:
            model_parameters: Iterable of model parameters to optimize.

        Returns:
            Optimizer: Configured optimizer instance.
        """
        return Adam(
            model_parameters,
            lr=self.learning_rate,
            weight_decay=self.weight_decay
        )

    def get_lr_scheduler(self, optimizer: Optimizer, num_epochs) -> LRScheduler | None:
        """
        Returns a PyTorch learning rate scheduler instance based on the hyperparameters.

        Args:
            optimizer: Optimizer instance for which to create the scheduler.
            num_epochs: Number of epochs used for cosine annealing.

        Returns:
            Optional[_LRScheduler]: Configured scheduler instance, or None if no scheduler is specified.
        """
        if self.lr_scheduler is None:
            return None
        elif self.lr_scheduler == "step":
            return StepLR(optimizer, step_size=self.lr_scheduler_step_size, gamma=self.lr_scheduler_gamma)
        elif self.lr_scheduler == "exp":
            return ExponentialLR(optimizer, gamma=self.lr_scheduler_gamma)
        elif self.lr_scheduler == "plateau":
            return ReduceLROnPlateau(optimizer, mode="min", factor=self.lr_scheduler_gamma, patience=5)
        elif self.lr_scheduler == "cosine":
            return CosineAnnealingLR(optimizer, T_max=num_epochs)
        else:
            raise ValueError(f"Unknown scheduler type: {self.lr_scheduler}")


class TrainingRequest(BaseModel):
    dataset_id: int = Field(default=1, description="Dataset ID")
    model_registry_key: str = Field(default="unet", description="A key from the model registry")
    hyper_params: HyperParams = Field(default_factory=HyperParams, description="Hyperparameters")
    augmentations: Augmentations = Field(default_factory=Augmentations, description="Augmentations")
    data_profile: DataProfile = Field(default_factory=DataProfile, description="DataProfile")
    num_epochs: int = Field(default=1, description="Number of epochs to train.")

    @field_validator('dataset_id')
    def validate_dataset_id(cls, value):
        if str(value) not in os.listdir(DATA_PATH):
            raise ValueError(f"Dataset with ID {value} does not exist in the data path {DATA_PATH}."
                             f"Please make sure to upload the dataset first.")
        return value
