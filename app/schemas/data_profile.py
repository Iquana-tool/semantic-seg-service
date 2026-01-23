from pydantic import BaseModel, Field


class DataProfile(BaseModel):
    """
    Dataset profile class. Keeps track of some information about the datasets like number of classes.
    """
    classes_dict: dict[int, str] = Field(default={}, description="Dictionary mapping class ID to its name.")
    image_size: tuple[int, int] = Field(default=(512, 512), description="Images will be resized to this size before use.")

    n_train: int = Field(default=0, description="Number of training images.")
    n_val: int = Field(default=0, description="Number of validation images.")
