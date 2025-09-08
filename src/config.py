import os

class Config:
    DATASET_PATH = os.path.join("data", "PetImages")
    IMAGE_SIZE = 32
    TOTAL_IMAGES = 24998

    # Training parameters
    BATCH_SIZE = 128

    # Data parameters
    TRAINING_SPLIT = 0.1
    VALIDATION_SPLIT = 0.05
    TEST_SPLIT = 0.85
