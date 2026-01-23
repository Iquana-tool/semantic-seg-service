import os

import torch
from celery.exceptions import TaskRevokedError

from app.schemas.training_progress import TrainingProgress
from models.model_info import ModelInfo
from paths import TRAINED_MODEL_WEIGHTS_PATH, LOG_PATH
from training.dataloader import get_dataloader
from training.metrics import dice_coeff, iou_score


def train_model_logic(task, model, model_info: ModelInfo):
    # Import directories, make sure they exist before starting training
    log_dir = os.path.join(LOG_PATH, str(model_info.identifier_str))
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(TRAINED_MODEL_WEIGHTS_PATH, exist_ok=True)

    # Update the task status separately
    task.update_state(state='STARTED')

    # Init these vars here so you dont run into errors on error catching
    train_metrics, val_metrics, epoch, progress = None, None, None, None

    try:
        # Load the dataloaders
        train_loader, val_loader = get_dataloader(
            model_info.training_req.image_urls,
            model_info.training_req.mask_urls,
            batch_size=model_info.training_req.hyperparams.batch_size,
            augmentations=model_info.training_req.augmentations,
            num_workers=6
        )
        #
        val_available = val_loader is not None
        progress = TrainingProgress(
            total_epochs=model_info.training_req.num_epochs,
            monitor_type='val' if val_available else 'train',
        )

        # Set the device
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Loss, we could also make this configurable
        criterion = torch.nn.CrossEntropyLoss().to(device)

        # Load the model
        model.to(device)

        # Load the optimizer and LR Scheduler
        optimizer = model_info.training_req.hyperparams.get_optimizer(model.parameters())
        lr_scheduler = model_info.training_req.hyperparams.get_lr_scheduler(optimizer, progress.total_epochs)

        for epoch in range(progress.current_epoch, progress.total_epochs):
            task.update_state(state='PROGRESS',
                              meta={
                                  'epoch': epoch,
                                  'best_epoch': progress.best_epoch,
                                  'best_metrics': progress.monitor_best_metric,
                                  'total_epochs': progress.total_epochs,
                                  'percent': int((epoch / progress.total_epochs) * 100),
                                  'train_metrics': train_metrics,
                                  'val_metrics': val_metrics,
                              }
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
                torch.save(model, model_info.model_path())

            # Early stopping
            if epoch - progress.best_epoch > model_info.training_req.hyperparams.early_stopping_patience > 0:
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
            meta={
                'epoch': epoch,
                'best_epoch': progress.best_epoch,
                'best_metrics': progress.monitor_best_metric,
                'total_epochs': progress.total_epochs,
                'percent': int((epoch / progress.total_epochs) * 100),
                'train_metrics': train_metrics,
                'val_metrics': val_metrics,
            }
        )
        model_info.training_progress = progress
        model_info.save_to_disk()


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
