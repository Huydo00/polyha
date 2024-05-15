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


link_camera1 = "rtsp://admin:BNNNRU@192.168.1.12:554/onvif1"
cap = cv2.VideoCapture(1)


minW = 0.1 * cap.get(3)
minH = 0.1 * cap.get(4)


while True:
    # On camera
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

    # resize
    img = cv2.resize(img (854, 480))

    cv2.imshow('Check face', img)

    k = cv2.waitKey(10) & 0xFF  #"ESC exit"
    if k == 27:
        break


print("\n [INFO] Done")
cap.release()
cv2.destroyAllWindows()
