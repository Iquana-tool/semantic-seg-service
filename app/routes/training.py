from celery.result import AsyncResult
from fastapi import APIRouter, HTTPException

from app.schemas.data_profile import DataProfile
from app.schemas.training_progress import TrainingProgress
from app.schemas.training_request import TrainingRequest
from app.state import MODEL_REGISTRY
from celery_app import celery
from celery_tasks.training import train_model_task
from models.model_info import ModelInfo

router = APIRouter(prefix="/training", tags=["training"])

@router.post("/start")
async def start_training(req: TrainingRequest):
    model_info = MODEL_REGISTRY.get_model_info(req.model_registry_key)
    model_loader = MODEL_REGISTRY.get_model_loader(req.model_registry_key)
    if not model_info.is_base_model():
        return {
            "success": False,
            "message": f"Service only supports training of base models, but {model_info.identifier_str} was provided."
        }
    new_identifier = MODEL_REGISTRY.get_new_key()
    new_model_info = model_info.copy()
    new_model_info.identifier_str = new_identifier
    new_model_info.type = "trained"
    new_model_info.training_req = req
    task: AsyncResult = train_model_task.delay(model_loader.load_model(), new_model_info)
    return {
        "success": True,
        "task_id": task.id,
        "message": "Training task enqueued."
    }


@router.get("/tasks", description="Get all tasks")
async def get_tasks():
    """ Returns a list of all tasks. """
    i = celery.control.inspect()

    # These calls return a dict keyed by worker name: {'worker@host': [tasks]}
    active = i.active() or {}
    reserved = i.reserved() or {}
    scheduled = i.scheduled() or {}

    def extract_ids(worker_map):
        task_ids = []
        for worker_name, tasks in worker_map.items():
            for task in tasks:
                task_ids.append(task['id'])
        return task_ids

    return {
        "active": extract_ids(active),  # Currently running on GPU/CPU
        "reserved": extract_ids(reserved),  # In queue, waiting for a worker
        "scheduled": extract_ids(scheduled)  # Tasks with an ETA/countdown
    }


@router.delete("/tasks/{task_id}")
async def stop_training(task_id: str):
    """
    Stop a running or queued training task by its task_id.
    """
    celery.control.revoke(task_id, terminate=True)
    return {
        "success": True,
        "status": "Stopped",
        "message": f"Training task {task_id} has been stopped/canceled."
    }


@router.get("/tasks/{task_id}")
async def get_training_progress(task_id: str):
    """ Returns a progress report for the training task. """
    res = AsyncResult(task_id, app=celery)

    # In Celery, during 'PROGRESS' state, the metadata is in 'res.info'
    # If the task is finished, 'res.result' contains the return value.
    progress_data = res.info if isinstance(res.info, dict) else {"status": res.state}

    return {
        "task_id": task_id,
        "state": res.state,
        "data": progress_data
    }
