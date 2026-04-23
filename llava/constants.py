CONTROLLER_HEART_BEAT_EXPIRATION = 30
WORKER_HEART_BEAT_INTERVAL = 15

LOGDIR = "."

# Model Constants
IGNORE_INDEX = -100
IMAGE_TOKEN_INDEX = -200
DEFAULT_IMAGE_TOKEN = "<image>"
DEFAULT_IMAGE_PATCH_TOKEN = "<im_patch>"
DEFAULT_IM_START_TOKEN = "<im_start>"
DEFAULT_IM_END_TOKEN = "<im_end>"

# Temporal grounding (Phase 2): anchors + discrete time bins <t000>… (width grows with mm_num_time_bins)
DEFAULT_TIME_START_TOKEN = "<time_start>"
DEFAULT_TIME_END_TOKEN = "<time_end>"
DEFAULT_NUM_TIME_BINS = 1000
