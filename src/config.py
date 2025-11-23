import os

class Config:
    DATASET_PATH = os.path.join("data", "MyOwnImages")
    EXCEL_PATH = os.path.join("C:/Users/dimos/Downloads/Εύρεση εικόνων για τον έλεγχο του AI μοντέλου (Απαντήσεις).xlsx")
    CATS_FOLDER_NAME = "Cat"
    DOGS_FOLDER_NAME = "Dog"
    IMAGE_SIZE = 128
    TOTAL_IMAGES = 24998

    # Training parameters
    BATCH_SIZE = 128
    EARLY_STOPPING = True
    EARLY_STOPPING_PATIENCE = 5

    # Data parameters
    TRAINING_SPLIT = 0.01
    VALIDATION_SPLIT = 0.01
    TEST_SPLIT = 0.98

    # Logging parameters
    EXPERIMENTS_FOLDER = "experiments"

    # Plotting parameters
    MAX_PLOTS_PER_ROW = 3
    MAX_ROWS_PER_GRAPH = 3
    FIGURE_WIDTH = 9
    FIGURE_HEIGHT = 10
    TITLE_FONT_SIZE = 16
    TITLE_PADDING = 26
    SUBTITLE_FONT_SIZE = 10
    SUBTITLE_X = 0.5
    SUBTITLE_Y = 1.04
    LABEL_FONT_SIZE = 12
    LABEL_PADDING = 10

