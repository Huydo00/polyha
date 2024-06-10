from ultralytics import YOLO
import cv2
import os
import numpy as np
from PIL import Image

recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read('trainer/trainer.yml')
cascadePath = "haarcascade_frontalface_default.xml"
faceCascade = cv2.CascadeClassifier(cascadePath)


font = cv2.FONT_HERSHEY_SIMPLEX


id = 0


names = ['...','huy']

# load video
link_camera1 = "rtsp://admin:BNNNRU@192.168.1.13:554/onvif1"
cap = cv2.VideoCapture("rtsp://admin:BNNNRU@192.168.1.9:554/onvif1")

# load yolov8 model
model = YOLO('yolov8n-pose.pt')

# track objects
class_name = 0  # Person
results = model.track(cap, tracker="bytetrack.yaml", stream=True, classes=class_name, conf=0.3)

minW = 0.1 * cap.get(3)
minH = 0.1 * cap.get(4)


def faceid():
    # On camera
    ret, frame = cap.read()
    frame = cv2.resize(frame, (854, 480))123456

    frame = frame
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = faceCascade.detectMultiScale(
        gray,
        scaleFactor=1.2,
        minNeighbors=5,
        minSize=(int(minW), int(minH)),
    )

    for (x, y, w, h) in faces:

        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 128, 0), 2)

        id, confidence = recognizer.predict(gray[y:y + h, x:x + w])

        if (confidence < 100):
            id = names[id]
            confidence = "  {0}%".format(round(100 - confidence))
        else:
            id = "No name"
            confidence = "  {0}%".format(round(100 - confidence))

        cv2.putText(frame, str(id), (x + 5, y - 5), font, 1, (255, 0, 0), 2)
        cv2.putText(frame, str(confidence), (x + 5, y + h - 5), font, 1, (255, 255, 255), 2)

    cv2.imshow('Check face', frame)


while True:
    # read frames
    ret, frame = cap.read()
    if ret:
        # detect objects
        results = model(frame)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # cv2.rectangle
        # cv2.putText
        frame = results[0].plot()

        # resize
        frame = cv2.resize(frame, (854, 480))

        DP = results[0].numpy()
        if len(DP) != 0:
            # plot
            # results
            # export data
            # Object face
            faceid()
        # for result in results:
        #     boxes = result[0].boxes.numpy()
        #     for box in boxes:
        #
        #         print("id", box.id)
        #         print("class", box.cls)
        #         print("xyxy", box.xyxy)
        #         print("conf", box.conf)
        #         print("\n")

        # visualize
        cv2.imshow('YOLOV8', frame)
        cv2.waitKey(1)

        k = cv2.waitKey(10) & 0xFF  # "ESC exit"
        if k == 27:
            break

    else:
        cap = cv2.VideoCapture("rtsp://admin:BNNNRU@192.168.1.9:554/onvif1")

cap.release()
cv2.destroyAllWindows()