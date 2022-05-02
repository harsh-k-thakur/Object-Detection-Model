import os
import time
from tracemalloc import start
from turtle import st

import cv2
import tensorflow as tf
import numpy as np

from tensorflow.python.keras.utils.data_utils import get_file

np.random.seed(6)

class Detector:
    def __init__(self) -> None:
        '''
        This function is used to initialize the Detector
        There is nothing in the initalize method at this point
        '''
        pass

    def readClasses(self, classesFilePath):
        '''
        There are around 92 classes which coco model can pick up right out of the box.
        This classes are listed in the coco.names file under the data/label folder.

        This method is used to save those classes in a variable
        Which can be used in the later stage of the program

        We have initalized each class with a different color, 
        and so each bounding box is will have different color in order to easily classify.
        '''
        with open(classesFilePath, 'r') as f:
            self.classesList = f.read().splitlines()

        # Color list
        self.colorList = np.random.uniform(low=0, high=255, size=(len(self.classesList), 3))
        # print(len(self.classesList), len(self.colorList))

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

                cv2.line(image, (x_max, y_min), (x_max-line_width, y_min), class_color, thickness=5)
                cv2.line(image, (x_max, y_min), (x_max, y_min+line_width), class_color, thickness=5)

                cv2.line(image, (x_min, y_max), (x_min+line_width, y_max), class_color, thickness=5)
                cv2.line(image, (x_min, y_max), (x_min, y_max-line_width), class_color, thickness=5)

                cv2.line(image, (x_max, y_max), (x_max-line_width, y_max), class_color, thickness=5)
                cv2.line(image, (x_max, y_max), (x_max, y_max-line_width), class_color, thickness=5)

        return image

    def predict_image(self, image_path, threshold=0.5):
        image = cv2.imread(image_path)

        bbox_image = self.create_bounding_box(image)

        output_path = os.path.abspath("../Object-detection-model/data/images/output/" + self.model_name + ".jpg")
        cv2.imwrite(output_path, bbox_image)
        cv2.imshow("Result", bbox_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        return
    
    def predict_video(self, video_path, threshold=0.5):
        cap = cv2.VideoCapture(video_path)

        if cap.isOpened() == False:
            print("Error Opening the file...")
            return

        start_time = 0
        (success, image) = cap.read()

        while success:
            current_time = time.time()

            fps = 1/(current_time - start_time)
            start_time = current_time

            bbox_image = self.create_bounding_box(image, threshold)

            cv2.putText(bbox_image, "FPS: " + str(int(fps)), (20, 70), cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 0), thickness=2)
            cv2.imshow("Result", bbox_image)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break

            (success, image) = cap.read()

        cv2.destroyAllWindows()
        return