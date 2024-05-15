import cv2
import os
import numpy as np

link_camera1 = "rtsp://admin:BNNNRU@192.168.1.12:554/onvif1"
cap = cv2.VideoCapture(link_camera1)

face_detector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

face_id = input('\n nhap ID: ')
print("\n [INFO] Camera Start")
count = 0


while True:
    #On camera
    ret, img = cap.read()


    #conver img  GRAY
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_detector.detectMultiScale(gray, 1.3, 5)
    
    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
        count += 1

        cv2.imwrite("dataset/User." + str(face_id) + '.' + str(count) + ".jpg", gray[y:y + h, x:x + w])
        cv2.imshow('TIIM TECH', img)
    
    k = cv2.waitKey(100) & 0xFF
    if k == 27:
        break
    elif count >= 30:
        break
print("\n [INFO] Done")
cap.release()
cv2.destroyAllWindows()



# while True:
#     ret, frame = cap.read()
#     frame = cv2.resize(frame,(854,480))

#     gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#     faces = face_cascade.detectMultiScale(gray, 1.3, 5)

#     for (x, y, w, h) in faces:
#         cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 5)
#         roi_gray = gray[y:y + w, x:x + w]
#         roi_color = frame[y:y + h, x:x + w]
#         eyes = eye_cascade.detectMultiScale(roi_gray, 1.3, 5)

#         for (ex, ey, ew, eh) in eyes:
#             cv2.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 5)
#         break

#     cv2.imshow('frame', frame)
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break

# cap.release()
# cv2.destroyAllWindows()