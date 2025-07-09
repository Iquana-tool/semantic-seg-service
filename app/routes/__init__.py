from fastapi import APIRouter

router = APIRouter()


@router.get("/health")
async def get_health():
    return {"success": True, "message": "Automatic segmentation service is running"}
