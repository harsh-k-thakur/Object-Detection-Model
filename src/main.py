import os
from Detector import *

model_url = "http://download.tensorflow.org/models/object_detection/tf2/20200711/ssd_mobilenet_v2_fpnlite_320x320_coco17_tpu-8.tar.gz"
label_path = os.path.abspath("../Object-detection-model/data/coco.names")

detector = Detector()
detector.readClasses(label_path)
detector.download_model(model_url)