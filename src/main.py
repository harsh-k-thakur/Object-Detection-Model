import os

from Detector import *

# model_url = "http://download.tensorflow.org/models/object_detection/tf2/20200711/ssd_mobilenet_v2_fpnlite_320x320_coco17_tpu-8.tar.gz"
model_url = "http://download.tensorflow.org/models/object_detection/tf2/20200711/efficientdet_d4_coco17_tpu-32.tar.gz"
# model_url = "http://download.tensorflow.org/models/object_detection/tf2/20200711/faster_rcnn_resnet152_v1_1024x1024_coco17_tpu-8.tar.gz"
# model_url = "http://download.tensorflow.org/models/object_detection/tf2/20210210/centernet_mobilenetv2fpn_512x512_coco17_od.tar.gz"
label_path = os.path.abspath("../Object-detection-model/data/labels/coco.names")
image_path = os.path.abspath("../Object-detection-model/data/images/input/test_0.jpg")
video_path = os.path.abspath("../Object-detection-model/data/videos/video_0.mp4")

threshold = 0.5

detector = Detector()
detector.readClasses(label_path)
detector.download_model(model_url)
detector.load_model()
# detector.predict_image(image_path, threshold=threshold)
detector.predict_video(video_path, threshold=0.5)