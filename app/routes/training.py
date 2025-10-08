from fastapi import APIRouter, HTTPException
from app.schemas.training_request import TrainingRequest
from celery_tasks.training import train_model
from app.state import MODEL_REGISTRY
from app.schemas.training_run import TrainingRun, TrainingProgress, JobStatusEnum
from celery_app import celery


router = APIRouter(prefix="/training", tags=["training"])

@router.post("/start_training")
async def start_training(req: TrainingRequest):
    model_registry_entry = MODEL_REGISTRY[req.model_registry_key]
    if model_registry_entry.info.is_base_model():
        training_run = TrainingRun(
            dataset_identifier=req.dataset_identifier,
            hyperparams=req.hyper_params,
            augmentations=req.augmentations,
            data_profile=req.data_profile,
            progress=TrainingProgress(total_epochs=req.num_epochs)
        )
        model_registry_entry = MODEL_REGISTRY.register_new_model_from_base_model(req.model_registry_key)
        model_registry_entry.info.training_run = training_run

    training_run = model_registry_entry.info.training_run
    training_run.progress.total_epochs = training_run.progress.current_epoch + req.num_epochs
    training_run.update_data_profile(req.data_profile)
    training_run.update_hyperparams(req.hyper_params)
    training_run.update_augmentations(req.augmentations)

    # Update the training status
    training_run.set_status("training", JobStatusEnum.QUEUED, "Training is queued.")

    task = train_model.delay(req.model_dump(), req.model_registry_key)
    training_run.task_id = task.id

    return {
        "success": True,
        "model_id": req.model_registry_key,
        "task_id": task.id,
        "status": "In progress",
        "message": "Training started in the background."
    }


@router.post("/stop_training/{registry_key}")
async def stop_training(registry_key: str):
    """
    Stop a running or queued training task by its task_id.
    """
    try:
        task_id = MODEL_REGISTRY.get_task_id_of_model(registry_key)
        if task_id:
            celery.control.revoke(task_id, terminate=True)
            return {
                "success": True,
                "status": "Stopped",
                "message": f"Training task {task_id} has been stopped/canceled."
            }
        else:
            return {
                "success": False,
                "status": "Not running or queued",
                "message": f"There is no task running or queued for the given registry key {registry_key}"
            }
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to stop task: {str(e)}")
