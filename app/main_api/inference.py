import httpx
from paths import BACKEND_URL


async def post_mask(mask, mask_id):
    """ Post a new mask to the backend. """
    url = f"{BACKEND_URL}/masks/post_mask/{mask_id}"
    with httpx.Client() as client:
        pass

