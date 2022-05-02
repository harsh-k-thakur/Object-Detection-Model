from genericpath import exists
from inspect import ClassFoundException
from lib2to3.pgen2.grammar import opmap_raw
import os
import time

import cv2
import tensorflow as tf
import numpy as np

from tensorflow.python.keras.utils.data_utils import get_file

np.random.seed(53)

class Detector:
    def __init__(self) -> None:
        pass

    def readClasses(self, classesFilePath):
        with open(classesFilePath, 'r') as f:
            self.classesList = f.read().splitlines()

        # Color list
        self.colorList = np.random.uniform(low=0, high=255, size=(len(self.classesList), 3))
        print(len(self.classesList), len(self.colorList))

    def download_model(self, model_url):

        filename = os.path.basename(model_url)
        self.model_name = filename[:filename.index('.')]

        self.cache_dir = os.path.abspath("../Object-detection-model/pretrained_model/")
        os.makedirs(self.cache_dir, exist_ok=True)