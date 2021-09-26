## Importing Required Libraries
import cv2

## Creating a List of Class Labels
classNames= []
classFile = '/home/marisha/Object_Detection_In_Real-Time/coco.names.txt'
with open(classFile,'rt') as f:
    classNames = f.read().rstrip('\n').split('\n')    

## Paths to the YOLO weights and model configuration
configPath = '/home/marisha/Object_Detection_In_Real-Time/ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt'
weightsPath = '/home/marisha/Object_Detection_In_Real-Time/frozen_inference_graph.pb'

# Loading our SSD object detector trained on COCO dataset (91 classes)
model = cv2.dnn_DetectionModel(weightsPath,configPath)
model.setInputSize(320, 320)
model.setInputScale(1.0 / 127.5)
model.setInputMean((127.5, 127.5, 127.5))
model.setInputSwapRB(True)

## Object Detection using Webcam
capture = cv2.VideoCapture(0)
capture.set(3, 1280)
capture.set(4, 720)
capture.set(10, 150)
if not capture.isOpened():
    raise IOEror('!!Cannot open video!!')
    
font_scale = 2
font = cv2.FONT_HERSHEY_DUPLEX
    
while True:
    ret, frame=capture.read()
    ClassIndex, confidence, bbox = model.detect(frame, confThreshold=0.45)
    
    if (len(ClassIndex)!=0):
        for classInd, conf, boxes in zip(ClassIndex.flatten(), confidence.flatten(), bbox):
            if classInd<=91 :
                cv2.rectangle(frame, boxes, (100,125,255), 4)
                cv2.putText(frame, classNames[classInd-1], (boxes[0]+10, boxes[1]+40), font, fontScale=font_scale, color=(0,240,0))
    
    cv2.imshow('Object Dectector--Press q to exit', frame)
    
    if cv2.waitKey(2) & 0xFF ==ord('q'):
        break

capture.release()
cv2.destroyAllWindows()
