from models import MODEL_REGISTRY
from enum import Enum
import json
from logging import getLogger


logger = getLogger(__name__)


class JobStatus(Enum):
    """ Enum for the job status. """
    IDLE = "idle"
    STARTING = "starting"
    IN_PROGRESS = "in progress"
    FINISHED = "finished"
    FAILED = "failed"
    STOPPED = "stopped"


class ModelInfo:
    """
    This class keeps track of the model information, such as the registry key and model ID. But also about the
    training status, such as whether the model is currently being trained or not or inference status.
    It allows easy saving and loading of the model information to/from a file.
    """
    def __init__(self, registry_key=None, job_id=None):

        # General info about the model
        self.model_identifier = registry_key
        self.job_id = job_id
        self.dataset_id = None
        if registry_key:
            base_model = MODEL_REGISTRY.get(registry_key)
            for k, v in base_model.items():
                if k != "getter":
                    setattr(self, k, v)
        else:
            self.Name = None
            self.Description = None


        # Status attributes
        self.training_status = JobStatus.IDLE
        self.inference_status = JobStatus.IDLE

        # Hyperparameters
        self.batch_size = None
        self.augment = None
        self.early_stopping = None
        self.lr = None
        self.num_classes = None
        self.in_channels = None
        self.image_size = None
        self.num_input_images = None

        # Training attributes
        self.epoch = 0
        self.best_epoch = -1
        self.total_epochs = 0
        self.train_dice = []
        self.train_iou = []
        self.train_loss = []
        self.val_dice = []
        self.val_iou = []
        self.val_loss = []
        self.test_dice = -1
        self.test_iou = -1
        self.best_train_dice = -1
        self.best_val_dice = -1
        self.best_test_dice = -1

    def update(self, data):
        """ Update the model information with a dictionary of data. """
        for key, value in data.items():
            if hasattr(self, key):
                setattr(self, key, value)
            else:
                raise ValueError(f"Invalid key '{key}' for ModelInfo.")

    def set_training_status(self, status):
        """ Set the training status of the model. """
        if isinstance(status, JobStatus):
            self.training_status = status
        else:
            try:
                self.training_status = JobStatus(status)
            except ValueError:
                raise ValueError(f"Invalid status '{status}'. Must be an instance of JobStatus Enum.")

    def set_inference_status(self, status):
        """ Set the inference status of the model. """
        if isinstance(status, JobStatus):
            self.inference_status = status
        else:
            try:
                self.inference_status = JobStatus(status)
            except ValueError:
                raise ValueError(f"Invalid inference status '{status}'. Must be an instance of JobStatus Enum.")

    def from_json_data(self, json_data):
        """ Load model information from a JSON object. """
        for key, value in json_data.items():
            if hasattr(self, key):
                setattr(self, key, value)
            else:
                logger.warning(f"Key '{key}' not found in ModelInfo. Adding anyways.")
                setattr(self, key, value)

    def to_dict(self):
        """ Convert model information to a JSON object. """
        data = {k: v for k, v in self.__dict__.items() if not k.startswith('_')}
        data['training_status'] = self.training_status.value
        data['inference_status'] = self.inference_status.value
        return data

    def save(self, save_path):
        """ Save the model information to a JSON file. """
        with open(save_path, 'w') as f:
            json.dump(self.to_dict(), f, indent=4)

    def load(self, load_path):
        """ Load the model information from a JSON file. """
        with open(load_path, 'r') as f:
            data = json.load(f)
            self.from_json_data(data)
            if 'training_status' in data:
                self.set_training_status(data['training_status'])
            if 'inference_status' in data:
                self.set_inference_status(data['inference_status'])

    def training_step(self, train_loss, train_dice, train_iou, val_loss=None, val_dice=None, val_iou=None):
        """ Update training step information. """
        self.epoch += 1
        self.train_loss.append(train_loss)
        self.train_dice.append(train_dice)
        self.train_iou.append(train_iou)
        if val_loss is not None:
            self.val_loss.append(val_loss)
            self.val_dice.append(val_dice)
            self.val_iou.append(val_iou)

