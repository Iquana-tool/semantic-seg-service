import os
import uuid
import json
from fastapi import APIRouter, BackgroundTasks
from app.schemas.training import TrainingRequest
from training.dataloader import get_dataloader
from models import MODEL_REGISTRY
from paths import DATA_PATH, MODEL_PATH, LOG_PATH, JOBS_PATH
from logging import getLogger
import torch
from torch.utils.tensorboard import SummaryWriter
from training.metrics import dice_coeff, iou_score
from training.early_stopping import EarlyStopping
from models import load_model_from_checkpoint_path

router = APIRouter(prefix="/train", tags=["training"])
logger = getLogger(__name__)


def save_job_status(job_id, status: str, result: str = "", extra: dict = None):
    os.makedirs(JOBS_PATH, exist_ok=True)
    obj = {"status": status, "result": result}
    if extra:
        obj.update(extra)
    with open(os.path.join(JOBS_PATH, f"{job_id}.json"), "w") as f:
        json.dump(obj, f)


def read_job_status(job_id):
    try:
        with open(os.path.join(JOBS_PATH, f"{job_id}.json"), "r") as f:
            return json.load(f)
    except Exception:
        return None


@router.post("/start_training")
async def start_training(req: TrainingRequest, background_tasks: BackgroundTasks):
    job_id = req.model_id
    dataset_path = os.path.join(DATA_PATH, str(req.dataset_id))
    log_dir = os.path.join(LOG_PATH, job_id)
    model_save_path = os.path.join(MODEL_PATH, f"{req.model_identifier}_{job_id}.pt")
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(MODEL_PATH, exist_ok=True)
    os.makedirs(JOBS_PATH, exist_ok=True)

    save_job_status(job_id, "queued")

    def background_train_job():
        try:
            # Your request and path setup; assumes variables: req, dataset_path, job_id, log_dir, model_save_path
            train_loader, val_loader, test_loader = get_dataloader(
                dataset_path,
                batch_size=req.batch_size,
                augment=req.augment,
                image_size=req.image_size,
                split=True,
                val_ratio=0.1,
                test_ratio=0.1,
                min_samples_for_split=12,
                seed=42,
            )
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            criterion = torch.nn.CrossEntropyLoss()
            optimizer = torch.optim.Adam(model.parameters(), lr=req.lr)
            writer = SummaryWriter(log_dir=log_dir)

            start_epoch = 1
            if not req.restart and os.path.exists(model_save_path):
                model, checkpoint = load_model_from_checkpoint_path(model_save_path, device=device, eval_mode=False)
                optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
                start_epoch = checkpoint.get("epoch", 1)
                logger.info(f"Resuming training of {req.model_identifier} from epoch {start_epoch}.")
            elif not req.restart and not os.path.exists(model_save_path):
                logger.warning(f"Checkpoint {model_save_path} does not exist. Starting training from scratch.")
                model = MODEL_REGISTRY[req.model_identifier](num_classes=req.num_classes, in_channels=req.in_channels)
                model = model.to(device)

            best_dice, best_epoch = 0.0, 0
            early_stopping = EarlyStopping(patience=8)
            history = []

            for epoch in range(1, req.epochs + 1):
                train_loss, train_dice, train_iou = run_one_epoch(model, train_loader, optimizer, criterion, device,
                                                                  train=True)
                val_loss, val_dice, val_iou = run_one_epoch(model, val_loader, optimizer, criterion, device,
                                                            train=False)

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

                if val_dice > best_dice:
                    best_dice = val_dice
                    best_epoch = epoch
                    checkpoint_obj = {
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "epoch": epoch,
                    }
                    torch.save(checkpoint_obj, model_save_path)
                    # Save meta info
                    with open(model_save_path + ".json", "w") as f:
                        json.dump({
                            "model_identifier": req.model_identifier,
                            "num_classes": req.num_classes,
                            "in_channels": req.in_channels,
                            "image_size": req.image_size,
                            "epoch": epoch,
                            "job_id": job_id,
                            "dataset_id": req.dataset_id,
                            "best_val_dice": best_dice,
                        }, f, indent=2)
                if req.early_stopping and early_stopping.step(val_dice):
                    break

            writer.close()

            # Evaluate on test set if available
            test_dice, test_iou = None, None
            if test_loader is not None:
                # If using advanced checkpoint, need to load model_state_dict out of dict
                model = load_model_from_checkpoint_path(model_save_path, device=device, eval_model=True)
                test_dice, test_iou = run_test(model, test_loader, device)

            status_extra = {
                "history": history,
                "best_epoch": best_epoch,
                "best_val_dice": best_dice,
                "test_dice": test_dice,
                "test_iou": test_iou,
            }
            save_job_status(job_id, "completed", result=model_save_path, extra=status_extra)

        except Exception as e:
            save_job_status(job_id, "failed", result=str(e))

    background_tasks.add_task(background_train_job)
    return {"success": True,
            "job_id": job_id,
            "status": "in progress",
            "message": "Training started in the background."}


@router.get("/get_job_status/{model_id}")
async def get_job_status(model_id: str):
    status = read_job_status(model_id)
    if status is None:
        return {"error": "Job not found"}
    return status


@router.get("/download_model/{model_id}")
async def download_model(model_id: str):
    status = read_job_status(model_id)
    if status is None or status['status'] != 'completed':
        return {"error": "Model not available for download"}
    from fastapi.responses import FileResponse
    return FileResponse(status['result'], filename=f"{model_id}.pt")


def run_one_epoch(model, loader, optimizer, criterion, device, train=True):
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
        loss = criterion(outputs, masks)
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
