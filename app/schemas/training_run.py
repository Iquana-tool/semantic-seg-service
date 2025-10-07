from datetime import datetime
from enum import Enum
from functools import wraps
from typing import Literal
from collections import defaultdict

from pydantic import BaseModel, Field

from app.schemas.data_profile import DataProfile
from app.schemas.training_request import HyperParams
from app.schemas.augmentations import Augmentations
from models.model_info import logger


class JobStatusEnum(Enum):
    """ Enum for the job status. """
    IDLE = "idle"
    STARTING = "starting"
    IN_PROGRESS = "in_progress"
    FINISHED = "finished"
    FAILED = "failed"
    STOPPED = "stopped"


class ModelStatus(BaseModel):
    """ Describe the models current status. For example: If a model is used for inference and this process is currently
        ongoing its ModelStatus.type will be 'inference' and its status will be 'in_progress'.
    """
    type: Literal["training", "inference"] | None = Field(default=None, description="Type of the last job executed with this "
                                                                           "model. Either 'inference' or 'training'.")
    status: JobStatusEnum = Field(default=JobStatusEnum.IDLE, description="The status of the job.")
    last_started_at: datetime | None = Field(default=None, description="The last time the job was started.")
    last_finished_at: datetime | None = Field(default=None, description="The last time the job was finished.")
    message: str | None = Field(default=None, description="A status message to display, eg for failure reasons.")

    def set_type(self, type: Literal["training", "inference"]):
        if self.type == type:
            # Type already set
            return
        if self.status in [JobStatusEnum.STARTING, JobStatusEnum.IN_PROGRESS]:
            raise ValueError("Cannot switch type while previous type task is still ongoing!")
        self.type = type

    def set_status(self, status: JobStatusEnum, time: datetime, msg: str | None) -> None:
        match status:
            case JobStatusEnum.IDLE:
                pass
            case JobStatusEnum.STARTING:
                self.start(time)
            case JobStatusEnum.IN_PROGRESS:
                self.status = JobStatusEnum.IN_PROGRESS
            case JobStatusEnum.FINISHED:
                self.finish(time)
            case JobStatusEnum.FAILED:
                self.failed(time)
            case JobStatusEnum.STOPPED:
                self.stop(time)
            case _:
                raise ValueError("Unknown job status: {}".format(status))
        self.message = msg

    def start(self, time: datetime) -> None:
        self.last_started_at = time
        self.last_finished_at = None
        self.status =JobStatusEnum.STARTING

    def finish(self, time: datetime) -> None:
        self.last_finished_at = time
        self.status = JobStatusEnum.FINISHED

    def stop(self, time: datetime) -> None:
        self.last_finished_at = time
        self.status = JobStatusEnum.STOPPED

    def failed(self, time: datetime) -> None:
        self.last_finished_at = time
        self.status = JobStatusEnum.FAILED


class RunIdentity(BaseModel):
    model_identifier: str = Field(..., description="Unique string identifying the model.")
    dataset_identifier: int = Field(..., description="Unique string identifying the dataset.")
    created_at: datetime = Field(default=datetime.now(), description="Date and time the job was created.")
    updated_at: datetime = Field(default=datetime.now(), description="Date and time the job was updated.")


class Metrics(BaseModel):
    """ Tracks an arbitrary number of metrics per epoch."""
    metrics: dict[str, list] = Field(default=defaultdict(list),
                                     description="Dictionary of metrics such as loss, accuracy, etc. Maps the metric "
                                                 "name to its values per epoch.")

    def add_metric(self, name, value):
        self.metrics[name].append(value)

    def add_metrics(self, metrics_dict: dict):
        for k, v in metrics_dict.items():
            self.add_metric(k, v)

    def __getattr__(self, item):
        return self.metrics[item]


class TrainingProgress(BaseModel):
    total_epochs: int = Field(..., description="Total number of epochs.")
    current_epoch: int = Field(default=-1, description="Current epoch.")
    best_epoch: int = Field(default=-1, description="Best epoch.")
    monitor_type: Literal["train", "val"] = Field(default="val", description="Whether to monitor train or validation "
                                                                             "values for figuring out best epoch.")
    monitor_metric: str = Field(default="loss", description="Metric to monitor for figuring out best epoch.")
    monitor_lower_is_better: bool = Field(default=True, description="Whether the metric gets better with decreasing (True)"
                                                                    "or increasing (False) values.")
    monitor_best_metric: float = Field(default=None, description="Value of the best epoch metric.")
    train_metrics: Metrics = Field(default_factory=Metrics, description="List of training metrics. Tracks metrics over epochs.")
    val_metrics: Metrics = Field(default_factory=Metrics, description="List of validation metrics. Tracks metrics over epochs.")

    def training_step(self, train_metrics, val_metrics=None) -> bool:
        """ Update training step information. Returns true if the new epoch was better than the best epoch yet,
        else false. """
        self.current_epoch += 1

        self.progress.train_metrics.add_metrics(train_metrics)
        if val_metrics is not None:
            self.progress.val_metrics.add_metrics(val_metrics)
            if self.monitor_type == "val":
                raise ValueError("Best epoch determination needs val metrics, but none were given.")

        return self.check_if_better_epoch(val_metrics[self.monitor_metric] if self.monitor_type == "val" else train_metrics[self.monitor_metric])

    def check_if_better_epoch(self, new_metric_value):
        # Check the condition for updating the best epoch, aka is it greater or smaller than the best seen value
        condition = (
                (self.monitor_lower_is_better and new_metric_value < self.monitor_best_metric)
                or
                (not self.monitor_lower_is_better and new_metric_value > self.monitor_best_metric)
        )
        # Update if it is better or if we have not yet recorded a value.
        if self.monitor_best_metric is None or condition:
            self.monitor_best_metric = new_metric_value
            self.best_epoch = self.current_epoch
            return True
        else:
            return False


def update(func):
    @wraps(func)
    def wrapper(self, *args, **kwargs):
        result = func(self, *args, **kwargs)  # Call the original method
        self.run_identity.updated_at = datetime.now()  # Update timestamp
        return result
    return wrapper


class TrainingRun(BaseModel):
    """
    This class keeps track of a training run, such as the registry key and model ID. But also about the
    training status, such as whether the model is currently being trained or not or inference status.
    It allows easy saving and loading of the model information to/from a file.
    """
    run_identity: RunIdentity = Field(...,
                                      description="Class object to identify the run.")
    model_status: ModelStatus = Field(default_factory=ModelStatus,
                                      description="Class object to track the status of the model.")
    hyperparams: HyperParams = Field(default_factory=HyperParams,
                                     description="Class object to track hyperparameters.")
    augmentations: Augmentations = Field(default_factory=Augmentations,
                                         description="Class object to track augmentations.")
    data_profile: DataProfile = Field(...,
                                      description="Class object to track training progress.")
    progress: TrainingProgress = Field(...,
                                       description="Class object to track progress.")

    def set_status(self, type: Literal["training", "inference"], status, msg: str | None = None):
        """ Set the training status of the model. """
        self.status.set_type(type)
        self.status.set_status(status, datetime.now(), msg)

    def from_json_data(self, json_data):
        """ Load model information from a JSON object. """
        for key, value in json_data.items():
            if hasattr(self, key):
                setattr(self, key, value)
            else:
                logger.warning(f"Key '{key}' not found in ModelInfo. Adding anyways.")
                setattr(self, key, value)

    def save(self, save_path):
        """ Save the model information to a JSON file. """
        with open(save_path, 'w') as f:
            f.write(self.model.json())

    @update
    def update_total_epochs(self, total_epochs, add_to_current=True):
        if add_to_current:
            self.progress.total_epochs = self.progress.current_epoch + total_epochs
        else:
            self.progress.total_epochs = total_epochs

    @update
    def update_hyperparams(self, hyper_params: HyperParams):
        self.hyperparams = hyper_params

    @update
    def update_augmentations(self, augmentations: Augmentations):
        self.augmentations = augmentations

    @update
    def update_data_profile(self, data_profile):
        self.data_profile = data_profile
