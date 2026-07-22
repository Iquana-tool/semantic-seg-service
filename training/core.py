import json
import os
from pathlib import Path

import redis
import torch
from celery.exceptions import TaskRevokedError
from iquana_toolbox.schemas.training import TrainingProgress, SemanticTrainingRequest
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau

from app.state import MODEL_REGISTRY
from celery_app import celery_app
from paths import TRAINED_MODEL_WEIGHTS_PATH, TRAINED_MODEL_INFO_PATHS, REDIS_URL
from training.dataloader import get_dataloader
from training.metrics import dice_coeff, iou_score


@celery_app.task(name="semantic_segmentation.train_model", bind=True)
def train_model_logic(task, req):
    req = SemanticTrainingRequest.model_validate_json(req)
    # Load everything from disk and model registry
    model_info = MODEL_REGISTRY.get_model_info(req.model_registry_key)
    model_loader = MODEL_REGISTRY.get_model_loader(req.model_registry_key)
    if not model_info.trainable:
        return {
            "success": False,
            "message": f"Model {req.model_registry_key} not trainable."
        }
    if model_info.pretrained and not model_info.finetunable:
        return {
            "success": False,
            "message": f"Model {req.model_registry_key} already trained and not finetunable."
        }

    # Set new fields
    new_identifier = MODEL_REGISTRY.get_new_key()
    model_info = model_info.copy()
    model_info.registry_key = new_identifier
    model_info.label_hierarchy = req.label_hierarchy

    # Load the model
    model = model_loader.load_model(
        in_channels=3,
        classes=len(list(model_info.label_hierarchy.id_to_label_object.keys())) + 1
    )

    # Load redis client and task_id
    redis_client = redis.from_url(REDIS_URL + "/0", decode_responses=True)
    task_id = task.request.id

    # Make necessary directories
    os.makedirs(TRAINED_MODEL_WEIGHTS_PATH, exist_ok=True)
    os.makedirs(TRAINED_MODEL_INFO_PATHS, exist_ok=True)

    # Save these paths for later
    info_path = Path(str(os.path.join(TRAINED_MODEL_INFO_PATHS, model_info.registry_key + ".json")))
    model_path = Path(str(os.path.join(TRAINED_MODEL_WEIGHTS_PATH, model_info.registry_key + ".pth")))

    # Update the task status separately
    task.update_state(state='STARTED')

    # Init these vars here so you dont run into errors on error catching
    train_metrics, val_metrics, epoch, progress = None, None, None, None

    try:
        # Load the dataloaders
        train_loader, val_loader = get_dataloader(
            req.image_urls,
            req.mask_urls,
            batch_size=req.hyper_params.batch_size,
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

        # Load the loss
        match req.loss:
            case "cross_entropy":
                criterion = torch.nn.CrossEntropyLoss().to(device)
            case _:
                raise ValueError(f"Unknown loss type '{req.loss}'")

        # Send model to device
        model.to(device)

        # This could also be user-specified, but it requires a way more sophisticated frontend. Might be future work.
        # Load the optimizer and LR Scheduler
        optimizer = Adam(
            model.parameters(),
            lr=req.hyper_params.learning_rate,
        )
        lr_scheduler = ReduceLROnPlateau(optimizer)

        # Training run
        for epoch in range(progress.epoch_count, req.num_epochs):
            # 1. Update Celery State
            current_meta = progress.model_dump()
            task.update_state(state='PROGRESS', meta=current_meta)

            # 2. Publish to Redis (for streaming)
            # We add the state to the dict so the stream knows what's happening
            stream_payload = {"state": "PROGRESS", "data": current_meta}
            redis_client.publish(f"task_progress_{task_id}", json.dumps(stream_payload))

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
            if epoch - progress.best_epoch > req.hyper_params.early_stopping_patience > 0:
                break
        task.update_state(state='SUCCESS')
        redis_client.publish(f"task_progress_{task_id}",
                             json.dumps({"state": "SUCCESS", "data": progress.model_dump()}))
    except Exception as e:
        redis_client.publish(f"task_progress_{task_id}",
                             json.dumps({
                                 "state": "FAILED",
                                 "data": progress.model_dump() if progress is not None else None
                             }))
        raise e
    finally:
        # Finally save the model if we actually trained something
        # For this, check whether a model has been saved!
        if os.path.exists(model_path):
            model_info.progress = progress
            model_info.pretrained = True
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
    for imgs, masks in loader:
        imgs, masks = imgs.to(device), masks.to(device)
        if train:
            optimizer.zero_grad()
        outputs = model(imgs)
        loss = criterion(outputs.float(), masks.long())
        if train:
            loss.backward()
            optimizer.step()
        running_loss += loss.item()
        running_dice += dice_coeff(outputs, masks).item()
        running_iou += iou_score(outputs, masks).item()
        nbatches += 1
    n = max(nbatches, 1)
    return {"loss": running_loss / n, "dice": running_dice / n, "iou": running_iou / n}
