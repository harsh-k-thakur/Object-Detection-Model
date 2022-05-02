from genericpath import exists
from inspect import ClassFoundException
from lib2to3.pgen2.grammar import opmap_raw
import os
import time

import cv2
import tensorflow as tf
import numpy as np

from tensorflow.python.keras.utils.data_utils import get_file

np.random.seed(6)

class Detector:
    def __init__(self) -> None:
        pass

    def readClasses(self, classesFilePath):
        with open(classesFilePath, 'r') as f:
            self.classesList = f.read().splitlines()

        # Color list
        self.colorList = np.random.uniform(low=0, high=255, size=(len(self.classesList), 3))
        print(len(self.classesList), len(self.colorList))

        return

    def download_model(self, model_url):

        filename = os.path.basename(model_url)
        self.model_name = filename[:filename.index('.')]

        self.cache_dir = os.path.abspath("../Object-detection-model/pretrained_model/")
        os.makedirs(self.cache_dir, exist_ok=True)

        get_file(fname=filename, 
                origin=model_url, 
                cache_dir=self.cache_dir,
                cache_subdir="checkpoints", 
                extract=True)
        
        return 

    def load_model(self):
        print("Loading the Model", self.model_name)
        tf.keras.backend.clear_session()
        self.model = tf.saved_model.load(os.path.join(self.cache_dir, "checkpoints", self.model_name, "saved_model"))

        print("Model '" + self.model_name + "' loaded successfully...")

        return

    def create_bounding_box(self, image, threshold=0.5):
        input_tensor = cv2.cvtColor(image.copy(), cv2.COLOR_BGR2RGB)
        input_tensor = tf.convert_to_tensor(input_tensor, dtype=np.uint8)
        input_tensor = input_tensor[tf.newaxis, ...]

        detections = self.model(input_tensor)

        bboxes = detections['detection_boxes'][0].numpy()
        class_indexes = detections['detection_classes'][0].numpy().astype(np.int32)
        class_scores = detections['detection_scores'][0].numpy()

        img_h, img_w, img_c = image.shape

        bbox_index = tf.image.non_max_suppression(bboxes, class_scores, max_output_size=50, iou_threshold=threshold, score_threshold=0.5)

        if len(bbox_index) != 0:
            for i in bbox_index:
                bbox = tuple(bboxes[i].tolist())
                class_confidence = round(100*class_scores[i])
                class_index = class_indexes[i]

                class_label_text = self.classesList[class_index]
                class_color = self.colorList[class_index]

                display_text = '{}: {}%'.format(class_label_text, class_confidence)

                y_min, x_min, y_max, x_max = bbox
                x_min, x_max, y_min, y_max = (x_min*img_w, x_max*img_w, y_min*img_h, y_max*img_h)
                x_min, x_max, y_min, y_max =  int(x_min), int(x_max), int(y_min), int(y_max)

                cv2.rectangle(image, (x_min, y_min), (x_max, y_max), color=class_color, thickness=1)
                cv2.putText(image, display_text, (x_min, y_min-10), cv2.FONT_HERSHEY_PLAIN, 1, class_color, thickness=2)

                line_width = min(int((x_max - x_min)*0.2), int((y_max - y_min)*0.2))

                cv2.line(image, (x_min, y_min), (x_min+line_width, y_min), class_color, thickness=5)
                cv2.line(image, (x_min, y_min), (x_min, y_min+line_width), class_color, thickness=5)

        return image

    def predict_image(self, image_path, threshold=0.5):
        image = cv2.imread(image_path)

        bbox_image = self.create_bounding_box(image)

        cv2.imwrite(self.model_name + ".jpg", bbox_image)
        cv2.imshow("Result", bbox_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()