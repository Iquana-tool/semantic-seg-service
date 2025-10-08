from pydantic import BaseModel, Field

from app.schemas.preprocessing import Preprocessing


class DataProfile(BaseModel):
    """
    Dataset profile class. Keeps track of some information about the datasets like number of classes and
    preprocessing. Additionally, makes sure that each run uses the same training and testing indices. This is very
    important, because otherwise we might switch previous training data into the test or validation set and then return
    misleading model performances.
    """
    num_samples: int = Field(default=0, description="Number of samples in the dataset.")

    num_classes: int = Field(default=0, description="Number of classes in the dataset.")
    classes_dict: dict[int, str] = Field(default_factory=dict, description="Dictionary mapping class ID to its name.")

    image_size: tuple[int, int] = Field(default=(256, 256), description="Images will be resized to this size before use.")
    preprocessing: Preprocessing = Field(default_factory=Preprocessing, description="Preprocessing used for training.")

    train_ratio: float = Field(default=0.8, description="Ratio of training samples in the dataset.")
    val_ratio: float = Field(default=0.1, description="Ratio of validation samples in the dataset.")
    test_ratio: float = Field(default=0.1, description="Ratio of test samples in the dataset.")
    train_files: list[str] = Field(default_factory=list, description="List of indices of training samples in the dataset.")
    val_files: list[str] = Field(default_factory=list, description="List of indices of validation samples in the dataset.")
    test_files: list[str] = Field(default_factory=list, description="List of indices of test samples in the dataset.")

    def is_compatible(self, other):
        return (
            self.num_classes == other.num_classes
            and
            self.classes_dict == other.classes_dict
            and
            self.image_size == other.image_size
        )
