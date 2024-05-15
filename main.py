from ultralytics import YOLO
import cv2
import numpy as np
from PIL import Image

recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read('trainer/trainer.yml')
cascadePath = "haarcascade_frontalface_default.xml"
faceCascade = cv2.CascadeClassifier(cascadePath)


font = cv2.FONT_HERSHEY_SIMPLEX


id = 0


names = ['...','huy']

# load yolov8 model
model = YOLO('yolov8n.pt')

# load video
video_path = 'cctv.mp4'
link_camera1 = "rtsp://admin:BNNNRU@192.168.1.12:554/onvif1"
cap = cv2.VideoCapture(1)

minW = 0.1 * cap.get(3)
minH = 0.1 * cap.get(4)

def id_face():
    ret, img = cap.read()
    img = img


    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


    faces = faceCascade.detectMultiScale(
        gray,
        scaleFactor=1.2,
        minNeighbors=5,
        minSize=(int(minW), int(minH)),
    )

    for (x, y, w, h) in faces:

        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)

        id, confidence = recognizer.predict(gray[y:y + h, x:x +w])

        if (confidence < 100):
            id = names[id]
            confidence = "  {0}%".format(round(100 - confidence))
        else:
            id = "No name"
            confidence = "  {0}%".format(round(100 - confidence))

        cv2.putText(img, str(id), (x + 5, y - 5), font, 1, (255, 255, 255), 2)
        cv2.putText(img, str(confidence), (x + 5, y + h - 5), font, 1, (255, 255, 255), 2)

    cv2.imshow('Check face', img)


# read frames
while True:
    ret, frame = cap.read()

    # detect objects

    # track objects
    results = model.track(frame, tracker="bytetrack.yaml", classes=0, conf=0.5)
    DP = results[0].numpy()

    if len(DP) != 0:
        # plot results
        # cv2.rectangle

        # cv2.putText
        frame = results[0].plot()

        id_face()

        # export data
        for result in results:
            boxes = result[0].boxes.numpy()
            for box in boxes:
                print("class", box.cls)
                print("xyxy", box.xyxy)
                print("conf", box.conf)


    # resize
    frame = cv2.resize(frame, (854, 480))

    # visualize
    cv2.imshow('YOLOV8', frame)


    k = cv2.waitKey(10) & 0xFF  # "ESC exit"
    if k == 27:
        break

cap.release()
cv2.destroyAllWindows()