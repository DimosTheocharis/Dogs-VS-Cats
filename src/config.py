import os

class Config:
    DATASET_PATH = os.path.join("data", "MyOwnImages")
    IMAGE_SIZE = 128
    TOTAL_IMAGES = 24998

    # Training parameters
    BATCH_SIZE = 128
    EARLY_STOPPING = True
    EARLY_STOPPING_PATIENCE = 5

    # Data parameters
    TRAINING_SPLIT = 0.0
    VALIDATION_SPLIT = 0.0
    TEST_SPLIT = 1.0

    # Logging parameters
    EXPERIMENTS_FOLDER = "experiments"

    # Plotting parameters
    MAX_PLOTS_PER_ROW = 3
    MAX_ROWS_PER_GRAPH = 3
    FIGURE_WIDTH = 9
    FIGURE_HEIGHT = 7

