# Object_Detection
Object detection in images and in real time videos.<br>
## Object Detection in Images
To Detect objects in images, I have used the famous YOLO algorithm ("You Only Look Once"). The algorithm works by framing object detection in images as a regression problem to spatially separated bounding boxes and associated class probabilities. In this approach, a single neural network divides the image into regions and predicts bounding boxes and probabilities for each region. The neural network predicts bounding boxes and class probabilities directly from full images in one evaluation. The base YOLO model processes images in real-time at 45 frames per second. To read more, refer to the paper [You Only Look Once: Unified, Real-Time Object Detection](https://arxiv.org/pdf/1506.02640v5.pdf).<br>
Here I have used **YOLOv3** algorithm. For reference: [YOLOv3: An Incremental Improvement](https://pjreddie.com/media/files/papers/YOLOv3.pdf) and [Github](https://github.com/pjreddie/darknet).<br><br>
**Important Libraries**<br>
*numpy* and *opencv* are the main libraries in the code. If you are running the code make sure you have these installed or simply use these commands in the command prompt:
```
pip install numpy
```
```
pip install opencv-python
```
**Important Files**<br>
Download the files and save it inside a folder and update the path to these files in the code. Folowing files have been used during training the model:<br>
[yolo.coco.names.txt](https://github.com/Marisha18/Object_Detection/blob/main/yolo.coco.names.txt)<br>
[yolov3.cfg.txt](https://github.com/Marisha18/Object_Detection/blob/main/yolov3.cfg.txt)<br>
[yolov3.weights](https://pjreddie.com/media/files/yolov3.weights)<br><br>
**Running The Code**<br>
You can now run the file by giving a similar command on your command prompt:
```
python yolo.py --image images/ipl.jpeg
```
You can use any image you want after the ```--image``` (or ```-i```) argument. Make sure you give the right path.<br>
Press **q** to quit the window of the image showing object detection.<br>
## Object Detection in Real Time Videos
Object Detection Using **SSD MobileNet**. The mobilenet-ssd model is a Single-Shot multibox Detection (SSD) network, intended to perform object detection. In simple words this file is a pre-trained Tensorflow model and has already been trained on the COCO dataset. Freezing is the process to identify and save all of required things (graph, weights etc) in a single file that you can easily use, but cannot be trained anymore. Frozen graphs are commonly used for inference in TensorFlow and are stepping stones for inference for other frameworks. Here I used files which would provide the path to configuration and weights of the model and provided them as arguments to cv2.dnn_DetectionModel(). The cv2.dnn_DetectionModel() is a deep neural network model that helps in loading a pretrained model, (ssd-mobilenet in our case). The DNN module allows loading pre-trained models of most popular deep learning frameworks. To read more, refer to the paper [SSD: Single Shot MultiBox Detector](https://arxiv.org/pdf/1512.02325.pdf).<br><br>
**Important Libraries**<br>
*opencv* is the main library in the code. If you are running the code make sure you have it installed.<br><br>
**Important Files**<br>
Download the files and save it inside a folder and update the path to these files in the code. Folowing files have been used during training the model:<br>
[ssd.coco.names.txt](https://github.com/Marisha18/Object_Detection/blob/main/ssd.coco.names.txt)<br>
[ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt](https://github.com/Marisha18/Object_Detection/blob/main/ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt)<br>
[frozen_inference_graph.pb](https://github.com/Marisha18/Object_Detection/blob/main/frozen_inference_graph.pb)<br><br>
<b>Running The Code</b><br>
You can now run the file by giving this command on your command prompt:
```
python ssd.py
```
Press **q** to quit the window of the image showing object detection.<br>
___________________________________________________
