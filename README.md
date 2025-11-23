
# Introduction

This project is a deep learning model that has learned to distinguish dog images from cat images.

The actual model is a Convolutional Neural Network (CNN) built with PyTorch that has been trained, validated, and tested in a dataset containing 25000 128x128  RGB total images of dogs and cats.

**Dataset**:  https://www.kaggle.com/datasets/shaunthesheep/microsoft-catsvsdogs-dataset

Actually, I have built many models with different architectures (number of layers, hyperparameters etc).

# Best model

https://github.com/DimosTheocharis/Dogs-VS-Cats/tree/main/experiments/experiment_04%202025-09-20%2021.31

 - Accuracy on training set: 96.46% (22499 images)
 - Accuracy on validation set: 94.08% (1250 images)
 - Accuracy on test set: 88.04% (1250 images)

Apart from CNN models, the project contains many utility functions that helped me find the best model. 

# Developer settings ([config.py](https://github.com/DimosTheocharis/Dogs-VS-Cats/blob/main/src/config.py))

The file `src/config.py`  gathers the settings for every in the project, in one place.

 - **DATASET_PATH**  =  os.path.join("data", "MyOwnImages") -> The path of the folder inside the project that contains the dog and cat images
 - **CATS_FOLDER_NAME**  =  "Cat" -> The name of the folder under *DATASET_PATH* directory, where the models will retrieve the cat images from.
 - **DOGS_FOLDER_NAME**  =  "Dog" -> The name of the folder under *DATASET_PATH* directory, where the models will retrieve the dog images from.
 - **IMAGE_SIZE**  =  128 -> Images will be resized to 128x128
 
 #### Training parameters
 
 - **BATCH_SIZE**  =  128 -> How many images the model will see each time before updating its weights
 - **EARLY_STOPPING**  =  True -> Whether or not the training should stop before reaching the specified epochs
 - **EARLY_STOPPING_PATIENCE**  =  5 -> The model will stop the training if in 5 consecutive epochs the validation loss isn't getting lower.
 
 #### Data parameters

- **TRAINING_SPLIT**  =  0.90 -> How many images (in percentage of the entire dataset) will be used for training.
 - **VALIDATION_SPLIT**  =  0.05 -> Same for validation
 - **TEST**  =  0.05 -> Same for test
 
#### Logging parameters

 - **EXPERIMENTS_FOLDER** =  "experiments" -> The directory of the project where the experiments will be saved
 
 #### Plotting parameters 
 These parameters concern the presentation of the results of a model's test.
 - **MAX_PLOTS_PER_ROW** = 3 -> The number of images per row
 - **MAX_ROWS_PER_GRAPH**  =  3 -> The number of rows per graph
 - **FIGURE_WIDTH**  =  9 -> The width of each graph in inches
 - **FIGURE_HEIGHT** = 10 -> The height of each graph in inches
 - **TITLE_FONT_SIZE**  =  16 -> The font size of graph's title (the name of the pet actually)
 - **TITLE_PADDING** = 26 -> The gap between graph's title and image grid
 - **SUBTITLE_FONT_SIZE**  =  10 -> The font size of graph's subtitle (the location of the pet actually)


# How to download

#### Step 0: Download dataset
Get dataset from https://github.com/DimosTheocharis/Dogs-VS-Cats and extract the contents inside the directory specified in the `Config.DATASET_PATH`

For example in my case I have the following structure:

project root

| -- data 

| -- -- PetImages

| -- -- -- Dog

| -- -- -- -- 0.jpg

| -- -- -- -- 1.jpg

| -- -- -- -- ...

| -- -- -- Cat

| -- -- -- -- 0.jpg

| -- -- -- -- 1.jpg

| -- -- -- -- ...

#### Step 1: Clone project

    git clone https://github.com/DimosTheocharis/Dogs-VS-Cats.git

#### Step 2: Install dependencies

    pip install -r requirements.txt

#### Step 3: Run notebook

    python -m notebook

#### Step 4: Open notebook
Go to 

    http://localhost:8888/tree

#### Step 5: Open main.ipynb

Open **main.ipynb** file

#### Step 6: Run any code block you want
Check below for what you can do with main.ipynb file.



# Scenarios (What can you do with this project)

**Note**:  Before running any of the experiments, you should run the code blockes in the *Setup* module:

    %load_ext autoreload
    %autoreload 2

    import  sys, os
    sys.path.append(os.path.abspath("..")) # add parent directory (project root)

## Scenario 1: Test model in your own data

#### Step 1) Select a model from *experiments* folder

For example I chose the best model (*experiments/experiment_04 2025-09-20 21.31*)

#### Step 2) Put your own images in a folder inside the project

For example I put some dog and cat images in the directory data/MyOwnImages

#### Step 3) Set config parameters

    DATASET_PATH  =  os.path.join("data", "MyOwnImages")


#### Step 4) Run the scenario code

    
    from  src.utils.data  import  loadData
    from  src.config  import  Config
    from  src.cnn  import  ConvolutionalNeuralNetwork
    from  src.architectures.doubleBarrelRegularized  import  DoubleBarrelRegularizedArchitecture
    from  src.utils.basics  import  loadModel
    from  src.test  import  testModel
    
    trainLoader, validationLoader, testLoader  =  loadData(Config.DATASET_PATH, batchSize=Config.BATCH_SIZE, dataAugmentationTechniques=[])
    
    model=ConvolutionalNeuralNetwork(
	    architecture=DoubleBarrelRegularizedArchitecture()
    )
    
    loadModel(model, "experiments/experiment_04 2025-09-20 21.31/model.pth")
    
    testModel(model, testLoader=testLoader)
    
Expect to see how many images the model classifier correctly, and how much confident it was:

 ![image](https://github.com/DimosTheocharis/Dogs-VS-Cats/blob/main/testResults/nine_predictions.png)






