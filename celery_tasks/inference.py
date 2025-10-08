import torch
from celery import shared_task

from app.schemas.training_run import JobStatusEnum
from app.state import MODEL_REGISTRY
from app.main_api.inference import post_mask


@shared_task(bind=True)
async def inference(self, images: dict[int, torch.Tensor], registry_entry):
    try:
        registry_entry.info.training_run.set_status("inference", JobStatusEnum.STARTING, "Starting inference")
        model = registry_entry.loader.load_model()
        model.to("cuda" if torch.cuda.is_available() else "cpu")
        model.eval()

        registry_entry.info.training_run.set_status("inference", JobStatusEnum.IN_PROGRESS, "Running inference")
        failed = []
        for image_id, image in images.items():
            try:
                outputs = model(image)
                mask = outputs.cpu().detach().numpy()
                await post_mask(image_id, mask)
            except Exception as e:
                failed.append(image_id)

        registry_entry.info.training_run.set_status("inference",
                                                    JobStatusEnum.FINISHED,
                                                    f"Finished inference. {len(failed)} failed images: {failed}")
    except Exception as e:
        registry_entry.info.training_run.set_status("inference",
                                                    JobStatusEnum.FAILED,
                                                    f"Failed inference. Error: {e}")