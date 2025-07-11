import os
from paths import MODEL_PATH
from models import parse_weight_file_name


def get_new_job_id():
    model_ids = sorted([parse_weight_file_name(filename)[1] for filename in os.listdir(MODEL_PATH)])
    if not model_ids:
        return 0
    prev_id = model_ids[0]
    if prev_id > 0:
        # The first id is larger than 0 so we can give this id back
        return 0
    elif model_ids[-1] == len(model_ids):
        # The last given id matches the length of the array meaning the entire array is filled continuously
        return model_ids[-1] + 1
    else:
        for model_id in model_ids:
            if prev_id + 1 != model_id:
                # The next id is larger than the last one plus one, so we can give this back
                return prev_id + 1
            else:
                prev_id = model_id
        else:
            # The entire model_ids array is filled from 0 to the end with no spaces, we give back a new id.
            # This shouldnt happen as we check this edge case before.
            return prev_id + 1
