import pandas as pd
from src.config import Config
from PIL import Image, ImageOps
import gdown
import os

try:
    df = pd.read_excel(Config.EXCEL_PATH)
    first = next(df.iterrows())

    for row in df.iterrows():
        # Extract form's fields
        imagePath: str = row[1]['1. Παρακαλώ επισυνάψτε μια φωτογραφία του σκύλου ή της γάτας σας.']
        animal: str = row[1]['2. Η φωτογραφία που ανεβάσατε, δείχνει σκύλο ή γάτα;']
        name: str = row[1]['3. Πώς λέγεται το ζωάκι;']
        location: str = row[1]['4. Πού ζει το ζωάκι;']

        # Combine animal's name with location if the location exists
        fileName: str = (f"{name}___{location}" if not pd.isnull(location) else name) + ".jpg"

        saveImagePath: str | None = None
        if (animal == 'Σκύλος'):
            saveImagePath = os.path.join(Config.DATASET_PATH, Config.DOGS_FOLDER_NAME, fileName)

        elif (animal == 'Γάτα'):
            saveImagePath = os.path.join(Config.DATASET_PATH, Config.CATS_FOLDER_NAME, fileName)


        tries = 1
        # Check if a file with the same name already exists (maybe there are more than 1 dogs with the same name in the same city)
        # If Λούνα___Θεσσαλονίκη.jpg already exists, try with Λούνα (1)___Θεσσαλονίκη.jpg
        while (os.path.exists(saveImagePath)):
            fileName: str = (f"{name} ({tries})___{location}" if not pd.isnull(location) else name) + ".jpg"
            if (animal == 'Σκύλος'):
                saveImagePath = os.path.join(Config.DATASET_PATH, Config.DOGS_FOLDER_NAME, fileName)

            elif (animal == 'Γάτα'):
                saveImagePath = os.path.join(Config.DATASET_PATH, Config.CATS_FOLDER_NAME, fileName)

            tries += 1
                

        if (saveImagePath):
            # Download the image
            gdown.download(imagePath, saveImagePath, fuzzy=True)
            
            # Remove any exif tags in order for the image to not be rotated when plotted
            im = Image.open(saveImagePath)
            im = ImageOps.exif_transpose(im)
            im.save(saveImagePath)

    

except Exception as e:
    print("Beware, an error occured!")
    print(e)