import numpy as np

# Parameters
BATCH_SIZE=32
DATASET='cars196'
IMAGE_SIZE=227
EMBEDDING_SIZE=64
LR_init=7e-5
LR_gen=1e-2
LR_s=1e-3
RECORD_STEP=20
NUM_EPOCHS_PHASE1=1
NUM_EPOCHS_PHASE2=5

# To approximately reduce the mean of input images
image_mean = np.array([123, 117, 104], dtype=np.float32)  # RGB
# To shape the array image_mean to (1, 1, 1, 3) => three channels
image_mean = image_mean[None, None, None, [2, 1, 0]]
neighbours = [1, 2, 4, 8, 16, 32]