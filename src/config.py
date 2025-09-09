import os

class Config:
    DATASET_PATH = os.path.join("data", "PetImages")
    IMAGE_SIZE = 32
    TOTAL_IMAGES = 24998

    # Training parameters
    BATCH_SIZE = 128
    EARLY_STOPPING = True
    EARLY_STOPPING_PATIENCE = 1

    # Data parameters
    TRAINING_SPLIT = 0.01
    VALIDATION_SPLIT = 0.01
    TEST_SPLIT = 0.98

    # Logging parameters
    EXPERIMENTS_FOLDER = "tests"
