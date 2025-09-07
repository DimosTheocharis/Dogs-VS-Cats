from typing import List
import os
from src.utils.basics import getCurrentTimeRepresentation


class Logger(object):
    '''
        Provides logging functionalities.
        Use it to log the results of your models to a specified folder and file.
        
        Note: The {fileName} should NOT contain any extension
        because the data will be logged in .txt files!

        @param folderPath: str -> The path to the folder where the log files will be stored. (ie "logs/firstModel")
        If not existing, it will be created. 
        @param fileName: str -> The name of the log file. (ie "trainingResults")
        @param appendTimestamp: bool -> Whether or not to append a timestamp in the end of the file name.
    '''
    def __init__(self, folderPath: str, fileName: str, appendTimestamp: bool = True):
        self.logging: bool = True

        if not os.path.exists(folderPath):
            os.makedirs(folderPath)

        if (appendTimestamp):
            self.__filePath: str = folderPath + "/" + fileName + " " + getCurrentTimeRepresentation() + ".txt"
        else:
            self.__filePath: str = folderPath + "/" + fileName + ".txt"
        
    def logData(self, messages: List[str], printToConsole: bool = True) -> None:
        '''
            Logs data to the {self.__filePath}
            @messages: List[str] -> The data to be logged.
            @printToConsole: bool -> Whether or not to log data to the console too
        '''
        if (self.logging):
            with open(self.__filePath, "a") as file:
                file.write("\n")
                file.writelines(messages)
                file.write("\n")

        if (printToConsole):
            print("")
            print(f"File: {self.__filePath}")
            print(" ".join(messages))
            print("")

    def getFilePath(self) -> str:
        return self.__filePath

