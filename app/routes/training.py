import os
import shutil
import json
from typing import Literal

from fastapi import APIRouter, BackgroundTasks
from app.schemas.training_request import TrainingRequest
from models.model_loader import PathModelLoader
from models.model_registry import ModelRegistryEntry
from training.dataloader import get_dataloader
from app.state import MODEL_REGISTRY
from paths import DATA_PATH, MODEL_PATH, LOG_PATH, TRAINING_RUNS_PATH
from logging import getLogger
import torch
from torch.utils.tensorboard import SummaryWriter
from training.metrics import dice_coeff, iou_score
from training.early_stopping import EarlyStopping
from models import load_model_from_checkpoint_path
from app.schemas.training_run import JobStatusEnum, TrainingRun, TrainingProgress, RunIdentity
from app.schemas.data_profile import DataProfile

router = APIRouter(prefix="/training", tags=["training"])
logger = getLogger(__name__)


def save_job_status(job_id, status: str, result: str = "", extra: dict = None):
    os.makedirs(TRAINING_RUNS_PATH, exist_ok=True)
    obj = {"status": status, "result": result}
    if extra:
        obj.update(extra)
    with open(os.path.join(TRAINING_RUNS_PATH, f"{job_id}.json"), "w") as f:
        json.dump(obj, f)


def read_job_status(job_id):
    try:
        with open(os.path.join(TRAINING_RUNS_PATH, f"{job_id}.json"), "r") as f:
            return json.load(f)
    except Exception:
        return None


def get_training_run(model_registry_entry: ModelRegistryEntry,
                     req: TrainingRequest) -> TrainingRun:
    """ Gets or creates a TrainingRun object based on the given ModelRegistryEntry."""
    if model_registry_entry.info.is_base_model():
        model_registry_key = MODEL_REGISTRY.register_new_model_from_base_model(req.model_registry_key)

    else:
        # Model was trained before. We already have a training run object.
        training_run: TrainingRun = model_registry_entry.info.training_run

    return training_run


@router.post("/start_training")
async def start_training(req: TrainingRequest, background_tasks: BackgroundTasks):
    """ Start a training run for a specified model with a specified dataset id and training parameters."""
    logger.info(f"Received training request: {req}")

    # Get the model registry entry and update its training run
    model_registry_entry = MODEL_REGISTRY[req.model_registry_key]
    if model_registry_entry.info.is_base_model():
        # We need to create a new TrainingRun object and a new RegistryEntry and link them
        training_run: TrainingRun = TrainingRun(
            dataset_identifier=req.dataset_identifier,
            hyperparams=req.hyper_params,
            augmentations=req.augmentations,
            data_profile=req.data_profile,
            progress=TrainingProgress(total_epochs=req.num_epochs)
        )
        model_registry_entry = MODEL_REGISTRY.register_new_model_from_base_model(req.model_registry_key)
        model_registry_entry.info.training_run = training_run

    # Get the training run object
    training_run: TrainingRun = model_registry_entry.info.training_run

    # Update training run params
    training_run.progress.total_epochs = training_run.progress.current_epoch + req.num_epochs
    training_run.update_data_profile(req.data_profile)
    training_run.update_hyperparams(req.hyper_params)
    training_run.update_augmentations(req.augmentations)

    # For easier access to the fields
    hyperparams = training_run.hyperparams
    augmentations = training_run.augmentations
    data_profile = training_run.data_profile
    progress = training_run.progress
    model_registry_key = model_registry_entry.info.identifier_str

    log_dir = os.path.join(LOG_PATH, str(model_registry_key))
    dataset_path = os.path.join(DATA_PATH, str(req.dataset_id))
    model_save_path = os.path.join(MODEL_PATH, f"{model_registry_key}.pt")

    if os.path.exists(log_dir):
        # Restarting training removes the entire log dir
        logger.warning(f"MODEL {model_registry_key}:: Log directory already exists: {log_dir}. Overwriting logs.")
        shutil.rmtree(log_dir)
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(MODEL_PATH, exist_ok=True)
    os.makedirs(TRAINING_RUNS_PATH, exist_ok=True)

    training_run.set_status("training", JobStatusEnum.STARTING, "Training is starting...")

    def background_train_job():
        try:
            # Get the data loader
            train_loader, val_loader, test_loader = get_dataloader(
                dataset_path,
                batch_size=hyperparams.batch_size,
                augmentations=augmentations,
                data_profile=data_profile,
                num_workers=6
            )
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

            # Our loss could also be configurable
            criterion = torch.nn.CrossEntropyLoss()

            # Tensorboard Writer
            writer = SummaryWriter(log_dir=log_dir)

            logger.info(f"Model {model_registry_key}: Device {device}.")
            start_epoch = 1

            # Load the model
            model = model_registry_entry.loader.load_model()
            model.to(device)

            # Load the optimizer
            # TODO

            # Set availability flags -> Better readability
            val_available = val_loader is not None
            test_available = test_loader is not None

            logger.info(
                f"Model {model_registry_key}: Validation set available: {val_available}, Test set available: {test_available}")

            training_run.set_status("training", JobStatusEnum.IN_PROGRESS,
                                    "Model and data loaded. Commencing training.")
            for epoch in range(start_epoch, start_epoch + req.epochs):
                train_metrics = run_one_epoch(model,
                                              train_loader,
                                              optimizer,
                                              criterion,
                                              device,
                                              writer,
                                              epoch=epoch,
                                              train=True)
                writer_add_metrics("train", train_metrics, writer, epoch)
                if val_available:
                    val_metrics = run_one_epoch(model,
                                                val_loader,
                                                optimizer,
                                                criterion,
                                                device,
                                                writer,
                                                epoch=epoch,
                                                train=False)
                    writer_add_metrics("val", val_metrics, writer, epoch)
                else:
                    val_metrics = None

                # Pass the metrics to the progress tracker and check whether the new epoch was better than the best yet
                is_new_best_epoch = progress.training_step(train_metrics, val_metrics)
                logger.debug(f"Model {model_registry_key}: Epoch {epoch} / {start_epoch + req.epochs}. "
                             f"Validation: {val_metrics} \t Training: {train_metrics}")

                if is_new_best_epoch:
                    logger.info(f"Model {model_registry_key}: New best! Saving best model to {model_save_path}.")
                    torch.save(model, model_save_path)

                    if type(model_registry_entry.loader) is not type(PathModelLoader):
                        # We only need to update this loader once, and only here. We need to original model loading
                        # at the start of the script.
                        model_registry_entry.loader = PathModelLoader(path_to_model=model_save_path)
                    # Evaluate on test set if available and only for best models
                    if test_available:
                        test_metrics = run_test(model, test_loader, device, criterion)
                        progress.add_test_metrics(test_metrics)

                # Early stopping
                if epoch - progress.best_epoch > hyperparams.early_stopping_patience > 0:
                    break

            writer.close()

            progress.set_status("training", JobStatusEnum.FINISHED, f"Successfully trained {model_registry_key}")
            logger.info(f"Model {model_registry_key}: Completed.")

        except Exception as e:
            if type(e) == FileNotFoundError:
                # We cause this error on purpose to stop the background task
                progress.set_status("training", JobStatusEnum.STOPPED, f"Training was stopped.")
            else:
                # If there is another error, then the task failed
                progress.set_status("training", JobStatusEnum.FAILED, f"Training failed! Error: {e}")
            raise e

    background_tasks.add_task(background_train_job)
    return {"success": True,
            "model_id": model_registry_key,
            "status": "In progress",
            "message": "Training started in the background."}


@router.get("/get_job_status/{model_id}")
async def get_job_status(model_id: str):
    status = read_job_status(model_id)
    if status is None:
        return {"success": True, "message": "No job", "status": "No job"}
    return status


@router.get("/cancel_job/{job_id}")
async def cancel_job(job_id: int):
    logger.warning("THIS IS A WORKAROUND FOR CANCELLING JOBS!\nIt works by deleting the log directory which leads to an"
                   " error with tensorboard, which in turn stops the background task. Using this might lead to "
                   "unexpected behaviour.")
    log_dir = os.path.join(LOG_PATH, str(job_id))
    shutil.rmtree(log_dir, ignore_errors=True)
    #delete_model(job_id)
    return {"success": True,
            "message": f"Training of model {job_id} should be cancelled. This might take a while. "
                       f"Please check again in a few seconds."}


@router.get("/download_model/{model_id}")
async def download_model(model_id: str):
    status = read_job_status(model_id)
    if status is None or status['status'] != 'completed':
        return {"error": "Model not available for download"}
    from fastapi.responses import FileResponse
    return FileResponse(status['result'], filename=f"{model_id}.pt")


def run_one_epoch(model, loader, optimizer, criterion, device, writer: SummaryWriter, epoch, train=True):
    """ Run one training/validation epoch. Save intermediate results to a tensorboard writer."""
    if loader is None:
        return 0.0, 0.0, 0.0
    running_loss, running_dice, running_iou, nbatches = 0.0, 0.0, 0.0, 0
    if train:
        model.train()
    else:
        model.eval()
    added = False
    for imgs, masks in loader:
        imgs, masks = imgs.to(device), masks.to(device)
        if train:
            optimizer.zero_grad()
        outputs = model(imgs)
        loss = criterion(outputs, masks)
        if not added:
            writer.add_images(f"Inputs/{'train' if train else 'val'}", imgs, epoch, dataformats="NCHW")
            # Assume masks shape = [B, H, W], int (class indices)
            masks_vis = masks.unsqueeze(1).float() / masks.max().clamp(min=1)  # [B,1,H,W] float in 0-1
            writer.add_images(f"Targets/{'train' if train else 'val'}", masks_vis, epoch, dataformats="NCHW")
            pred_classes = torch.argmax(outputs, dim=1)  # [B,H,W]
            pred_vis = pred_classes.unsqueeze(1).float() / pred_classes.max().clamp(min=1)
            writer.add_images(f"Outputs/{'train' if train else 'val'}", pred_vis, epoch, dataformats="NCHW")
            added = True
        if train:
            loss.backward()
            optimizer.step()
        running_loss += loss.item()
        running_dice += dice_coeff(outputs, masks).item()
        running_iou += iou_score(outputs, masks).item()
        nbatches += 1
    n = max(nbatches, 1)
    return {"loss": running_loss / n, "dice": running_dice / n, "iou": running_iou / n}


def run_test(model, test_loader, device, criterion):
    test_loss, test_dice, test_iou, ntest = 0.0, 0.0, 0.0, 0
    with torch.no_grad():
        for imgs, masks in test_loader:
            imgs, masks = imgs.to(device), masks.to(device)
            outputs = model(imgs)
            test_loss += criterion(outputs, masks).item()
            test_dice += dice_coeff(outputs, masks).item()
            test_iou += iou_score(outputs, masks).item()
            ntest += 1
    n = max(ntest, 1)
    return {"loss": test_loss / n, "dice": test_dice / n, "iou": test_iou / n}


def writer_add_metrics(mode: Literal["train", "val", "test"], metrics: dict, writer: SummaryWriter, epoch: int):
    for metric, value in metrics.items():
        writer.add_scalar(f"{mode}/{metric}", value, epoch)
