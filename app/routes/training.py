import os
import shutil
import json
from fastapi import APIRouter, BackgroundTasks
from app.schemas.training_request import TrainingRequest
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
                    data_profile: DataProfile,
                    req: TrainingRequest) -> TrainingRun:
    """ Gets or creates a TrainingRun object based on the given ModelRegistryEntry."""
    if model_registry_entry.info.is_base_model():
        model_registry_key = MODEL_REGISTRY.get_available_new_key_for_base_model(req.model_registry_key)
        # We need to create a new TrainingRun object
        training_run: TrainingRun = TrainingRun(
            run_identity=RunIdentity(
                model_identifier=model_registry_key,
                dataset_identifier=req.dataset_identifier,
            ),
            hyperparams=req.hyper_params,
            augmentations=req.augmentations,
            data_profile=data_profile,
            progress=TrainingProgress(total_epochs=req.num_epochs)
        )
    else:
        # Model was trained before. We already have a training run object.
        training_run: TrainingRun = model_registry_entry.info.training_run

        # Check if the old model is compatible with the new data profile.
        if not data_profile.is_compatible(training_run.data_profile):
            error_msg = f"New data profile does not match data_profile of previous run. This can be" \
                        f" due to the changes in the classes or the image size. Note that these changes require new " \
                        f"models to be trained and old models cannot be retrained on the new data."
            logger.error(error_msg)
            raise Exception(error_msg)

        # Update the total_epochs by adding the current epoch to the total epochs.
        training_run.progress.total_epochs = training_run.progress.current_epoch + req.num_epochs
        # Update other fields
        training_run.update_data_profile(data_profile)
        training_run.update_hyperparams(req.hyper_params)
        training_run.update_augmentations(req.augmentations)
    return training_run



@router.post("/start_training")
async def start_training(req: TrainingRequest, background_tasks: BackgroundTasks):
    """ Start a training run for a specified model with a specified dataset id and training parameters."""
    logger.info(f"Received training request: {req}")
    # Load the data profile from disk
    dataset_path = os.path.join(DATA_PATH, str(req.dataset_id))
    data_profile = DataProfile.model_validate_json(os.path.join(dataset_path, "data_profile.json"))

    # Get the model registry entry and update its training run
    model_registry_entry = MODEL_REGISTRY[req.model_registry_key]
    model_registry_entry.info.training_run = get_training_run(model_registry_entry, data_profile, req)

    # For easier access to the fields
    training_run = model_registry_entry.info.training_run
    hyperparams = training_run.hyperparams
    augmentations = training_run.augmentations
    progress = training_run.progress
    model_registry_key = training_run.run_identity.model_identifier

    log_dir = os.path.join(LOG_PATH, str(model_registry_key))
    model_save_path = os.path.join(MODEL_PATH, f"{model_registry_key}.pt")
    run_save_path = os.path.join(TRAINING_RUNS_PATH, f"{model_registry_key}.json")
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
            # Your request and path setup; assumes variables: req, dataset_path, job_id, log_dir, model_save_path
            train_loader, val_loader, test_loader = get_dataloader(
                dataset_path,
                batch_size=hyperparams.batch_size,
                augmentations=augmentations,
                data_profile=data_profile,
                num_workers=6
            )
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            criterion = torch.nn.CrossEntropyLoss()
            writer = SummaryWriter(log_dir=log_dir)
            logger.info(f"Model {model_registry_key}: Device {device}.")

            start_epoch = 1
            if os.path.exists(model_save_path):
                model, checkpoint = load_model_from_checkpoint_path(model_save_path, device=device, eval_mode=False)
                optimizer = torch.optim.Adam(model.parameters(), lr=req.lr)
                optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
                start_epoch = checkpoint.get("epoch", 1)
                logger.info(f"JOB {job_id}: Resuming training of {req.model_identifier} from epoch {start_epoch}.")
            else:
                logger.warning(f"JOB {job_id}: Checkpoint {model_save_path} does not exist. Starting training from scratch.")
                model = (MODEL_REGISTRY[registry_key])["getter"](classes=req.num_classes, in_channels=req.in_channels)
                optimizer = torch.optim.Adam(model.parameters(), lr=req.lr)
                model = model.to(device)

            early_stopping = EarlyStopping(patience=8)
            history = []
            val_available = val_loader is not None
            test_available = test_loader is not None
            logger.info(f"JOB {job_id}: Validation set available: {val_available}, Test set available: {test_available}")
            val_loss, val_dice, val_iou = -1., -1., -1.
            save_job_status(job_id, "in progress")
            model_info.set_training_status(JobStatusEnum.IN_PROGRESS)
            model_info.num_input_images = len(train_loader.dataset)
            for epoch in range(start_epoch, start_epoch + req.epochs):
                train_loss, train_dice, train_iou = run_one_epoch(model,
                                                                  train_loader,
                                                                  optimizer,
                                                                  criterion,
                                                                  device,
                                                                  writer,
                                                                  epoch=epoch,
                                                                  train=True)
                if val_available:
                    val_loss, val_dice, val_iou = run_one_epoch(model,
                                                                val_loader,
                                                                optimizer,
                                                                criterion,
                                                                device,
                                                                writer,
                                                                epoch=epoch,
                                                                train=False)

                logger.debug(f"JOB {job_id}: Epoch {epoch} / {start_epoch + req.epochs}. "
                             f"Validation dice: {val_dice:.2%} \t Training dice: {train_dice:.2%}")
                writer.add_scalar("Loss/train", train_loss, epoch)
                writer.add_scalar("Loss/val", val_loss, epoch)
                writer.add_scalar("Dice/train", train_dice, epoch)
                writer.add_scalar("Dice/val", val_dice, epoch)
                writer.add_scalar("IoU/train", train_iou, epoch)
                writer.add_scalar("IoU/val", val_iou, epoch)

                history.append(dict(
                    epoch=epoch,
                    train_loss=train_loss, val_loss=val_loss,
                    train_dice=train_dice, val_dice=val_dice,
                    train_iou=train_iou, val_iou=val_iou
                ))
                metric_to_measure = val_dice if val_available else train_dice
                model_info.training_step(train_loss, train_dice, train_iou, val_loss, val_dice, val_iou)
                if ((val_available and val_dice > model_info.best_val_dice) or
                        (not val_available and train_dice > model_info.best_train_dice)):
                    logger.info(f"JOB {job_id}: Saving best model to {model_save_path}.")
                    if val_available:
                        model_info.best_val_dice = val_dice
                    model_info.best_train_dice = train_dice
                    model_info.best_epoch = epoch
                    checkpoint_obj = {
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "epoch": epoch,
                    }
                    torch.save(checkpoint_obj, model_save_path)
                    # Evaluate on test set if available and model improved
                    test_dice, test_iou = None, None
                    if test_available:
                        test_dice, test_iou = run_test(model, test_loader, device)
                        model_info.update(
                            {
                                "test_dice": test_dice,
                                "best_test_dice": test_dice,
                                "test_iou": test_iou,
                            }
                        )
                # Save meta info
                model_info.save(run_save_path)
                if epoch % 5 == 0:
                    status_extra = {
                        "epoch": epoch + 1,
                        "total_epochs": req.epochs,
                        "train_dice": train_dice,
                        "val_dice": val_dice,
                    }
                    save_job_status(job_id, "in progress", extra=status_extra)
                if req.early_stopping and early_stopping.step(metric_to_measure):
                    break

            writer.close()

            status_extra = {
                "history": history,
                "best_epoch": model_info.best_epoch,
                "best_val_dice": model_info.best_val_dice,
                "test_dice": model_info.test_dice if test_available else None,
                "test_iou": model_info.test_iou if test_available else None,
            }
            model_info.set_training_status(JobStatusEnum.FINISHED)
            # Save meta info
            model_info.save(run_save_path)
            logger.info(f"JOB {job_id}: Completed.")
            save_job_status(job_id, "completed", result=model_save_path, extra=status_extra)

        except Exception as e:
            if type(e) == FileNotFoundError:
                # We cause this error on purpose to stop the background task
                model_info.set_training_status(JobStatusEnum.STOPPED)
            else:
                # If there is another error, then the task failed
                model_info.set_training_status(JobStatusEnum.FAILED)
            # Save meta info
            model_info.save(run_save_path)
            save_job_status(job_id, "failed", result=str(e))
            raise e

    background_tasks.add_task(background_train_job)
    return {"success": True,
            "job_id": job_id,
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
        running_iou  += iou_score(outputs, masks).item()
        nbatches += 1
    n = max(nbatches, 1)
    return running_loss / n, running_dice / n, running_iou / n

def run_test(model, test_loader, device):
    test_dice, test_iou, ntest = 0.0, 0.0, 0
    with torch.no_grad():
        for imgs, masks in test_loader:
            imgs, masks = imgs.to(device), masks.to(device)
            outputs = model(imgs)
            test_dice += dice_coeff(outputs, masks).item()
            test_iou += iou_score(outputs, masks).item()
            ntest += 1
    n = max(ntest, 1)
    return test_dice / n, test_iou / n
