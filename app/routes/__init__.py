from fastapi import APIRouter
import torch

router = APIRouter()


@router.get("/health")
async def get_health():
    return {"success": True, "message": f"Automatic segmentation service is running on "
                                        f"{torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}"}
