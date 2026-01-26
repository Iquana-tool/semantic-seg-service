import os
from pathlib import Path

import torch
from celery.exceptions import TaskRevokedError

from schemas.training import TrainingProgress, SemanticTrainingRequest
from schemas.models import SemanticSegmentationModels
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau

from paths import TRAINED_MODEL_WEIGHTS_PATH, LOG_PATH, TRAINED_MODEL_INFO_PATHS
from training.dataloader import get_dataloader
from training.metrics import dice_coeff, iou_score


def train_model_logic(task, model, model_info: SemanticSegmentationModels, req: SemanticTrainingRequest):
    # Import directories, make sure they exist before starting training
    os.makedirs(TRAINED_MODEL_WEIGHTS_PATH, exist_ok=True)
    os.makedirs(TRAINED_MODEL_INFO_PATHS, exist_ok=True)
    info_path = Path(os.path.join(TRAINED_MODEL_INFO_PATHS, model_info.registry_key + ".json"))
    model_path = Path(os.path.join(TRAINED_MODEL_WEIGHTS_PATH, model_info.registry_key + ".pth"))

    # Update the task status separately
    task.update_state(state='STARTED')

    # Init these vars here so you dont run into errors on error catching
    train_metrics, val_metrics, epoch, progress = None, None, None, None


    try:
        # Load the dataloaders
        train_loader, val_loader = get_dataloader(
            req.image_urls,
            req.mask_urls,
            batch_size=req.hyperparams.batch_size,
            augmentations=req.augmentations,
            num_workers=6
        )
        #
        val_available = val_loader is not None
        progress = TrainingProgress(
            monitored_metric_type='val' if val_available else 'train',
        )

        # Set the device
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        match req.loss:
            case "cross_entropy":
                criterion = torch.nn.CrossEntropyLoss().to(device)
            case _:
                raise ValueError(f"Unknown loss type '{req.loss}'")

        # Load the model
        model.to(device)

        # This could also be user-specified, but it requires a way more sophisticated frontend. Might be future work.
        # Load the optimizer and LR Scheduler
        optimizer = Adam(
            model.parameters(),
            lr=req.hyper_params.learning_rate,
        )
        lr_scheduler = ReduceLROnPlateau(optimizer)

        for epoch in range(progress.current_epoch, req.num_epochs):
            task.update_state(state='PROGRESS',
                              meta=progress.model_dump()
                              )
            # Training round
            train_metrics = run_one_epoch(model, train_loader, optimizer, criterion, device, epoch, train=True)

            # Validation round
            if val_available:
                val_metrics = run_one_epoch(model, val_loader, optimizer, criterion, device, epoch, train=False)
            else:
                val_metrics = None

            # Update scheduler
            if lr_scheduler is not None:
                lr_scheduler.step(epoch)

            # Update the training progress and check whether we have a new best epoch
            is_new_best_epoch = progress.training_step(train_metrics, val_metrics)
            if is_new_best_epoch:
                # Save the model
                torch.save(model, model_path)

            # Early stopping
            if epoch - progress.best_epoch > req.hyperparams.early_stopping_patience > 0:
                break
        task.update_state(state='SUCCESS')
    except TaskRevokedError:
        task.update_state(state='STOPPED')
    except Exception as e:
        task.update_state(state='FAILURE', meta={'error': str(e)})
        raise e
    finally:
        # Close everything and save
        task.update_state(
            meta=progress.model_dump()
        )
        model_info.progress = progress
        info_path.write_text(model_info.model_dump_json(indent=4))

def run_one_epoch(model, loader, optimizer, criterion, device, epoch, train=True):
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
        if train:
            loss.backward()
            optimizer.step()
        running_loss += loss.item()
        running_dice += dice_coeff(outputs, masks).item()
        running_iou += iou_score(outputs, masks).item()
        nbatches += 1
    n = max(nbatches, 1)
    return {"loss": running_loss / n, "dice": running_dice / n, "iou": running_iou / n}
