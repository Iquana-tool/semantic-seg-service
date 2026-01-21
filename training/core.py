import os
from typing import Literal
import torch
from torch.utils.tensorboard import SummaryWriter
from app.state import MODEL_REGISTRY
from models.model_loader import PathModelLoader
from paths import DATA_PATH, MODEL_WEIGHTS_PATH, LOG_PATH, MODEL_REGISTRY_ENTRY_PATHS
from training.dataloader import get_dataloader
from training.metrics import dice_coeff, iou_score
from app.schemas.training_run import JobStatusEnum
from app.schemas.training_request import TrainingRequest
from celery.exceptions import TaskRevokedError


def train_model_logic(task, req_dict, model_registry_key):
    # Get the training request
    req = TrainingRequest.model_validate(req_dict)

    # Get the model registry entry. Note: This must exist as it is created before this task is called.
    model_registry_entry = MODEL_REGISTRY[model_registry_key]

    # Easy access to some fields, that we regularly update or use
    training_run = model_registry_entry.info.training_run
    hyperparams = training_run.hyperparams
    augmentations = training_run.augmentations
    data_profile = training_run.data_profile
    progress = training_run.progress

    # Import directories, make sure they exist before starting training
    log_dir = os.path.join(LOG_PATH, str(model_registry_key))
    dataset_path = os.path.join(DATA_PATH, str(req.dataset_id))
    model_save_path = os.path.join(MODEL_WEIGHTS_PATH, f"{model_registry_key}.pt")
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(MODEL_WEIGHTS_PATH, exist_ok=True)
    # Tensorboard logger
    writer = SummaryWriter(log_dir=log_dir)

    # Update the training status
    training_run.set_status("training", JobStatusEnum.STARTING, "Training is starting...")

    # Update the task status separately
    task.update_state(state='STARTED', meta={'status': 'Training started'})

    try:
        # Load the dataloaders
        train_loader, val_loader, test_loader = get_dataloader(
            dataset_path,
            batch_size=hyperparams.batch_size,
            augmentations=augmentations,
            data_profile=data_profile,
            num_workers=6
        )

        # Set the device
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Loss, we could also make this configurable
        criterion = torch.nn.CrossEntropyLoss()

        # Load the model
        model = model_registry_entry.loader.load_model()
        model.to(device)

        # Load the optimizer and LR Scheduler
        optimizer = hyperparams.get_optimizer(model.parameters())
        lr_scheduler = hyperparams.get_lr_scheduler(optimizer, progress.total_epochs)

        # Set some flags for readability
        val_available = val_loader is not None
        test_available = test_loader is not None

        # Training starts now, so update the status
        training_run.set_status("training", JobStatusEnum.IN_PROGRESS, "Model and data loaded. Commencing training.")
        task.update_state(state='PROGRESS', meta={'status': 'Training in progress'})

        for epoch in range(progress.current_epoch, progress.total_epochs):
            # Training round
            train_metrics = run_one_epoch(model, train_loader, optimizer, criterion, device, writer, epoch, train=True)
            writer_add_metrics("train", train_metrics, writer, epoch)

            # Validation round
            if val_available:
                val_metrics = run_one_epoch(model, val_loader, optimizer, criterion, device, writer, epoch, train=False)
                writer_add_metrics("val", val_metrics, writer, epoch)
            else:
                val_metrics = None

            # Update scheduler
            if lr_scheduler is not None:
                lr_scheduler.step(epoch)

            # Update the training progress and check whether we have a new best epoch
            is_new_best_epoch = progress.training_step(train_metrics, val_metrics)
            if is_new_best_epoch:
                # Save the model
                torch.save(model, model_save_path)
                if type(model_registry_entry.loader) is not type(PathModelLoader):
                    # Update the loader function. This happens exactly once.
                    model_registry_entry.loader = PathModelLoader(path_to_model=model_save_path)
                if test_available:
                    # Evaluate on a test set if available
                    test_metrics = run_test(model, test_loader, device, criterion)
                    progress.add_test_metrics(test_metrics)

            # Early stopping
            if epoch - progress.best_epoch > hyperparams.early_stopping_patience > 0:
                break
        progress.set_status("training", JobStatusEnum.FINISHED, f"Successfully trained {model_registry_key}")
        task.update_state(state='SUCCESS', meta={'status': 'Training completed'})
    except TaskRevokedError:
        # Handle task revoked separately
        progress.set_status("training", JobStatusEnum.STOPPED, "Training stopped.")
        task.update_state(state='STOPPED', meta={'status': f'Training stopped.'})
    except Exception as e:
        # Raise every other error
        progress.set_status("training", JobStatusEnum.FAILED, f"Training failed! Error: {e}")
        task.update_state(state='FAILURE', meta={'status': f'Training failed: {str(e)}'})
        raise e
    finally:
        # Close everything and save
        writer.close()
        model_registry_entry.info.training_run.task_id = None
        entry_path = os.path.join(MODEL_REGISTRY_ENTRY_PATHS, f"{model_registry_key}.json")
        model_registry_entry.save(entry_path)

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