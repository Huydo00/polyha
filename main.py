# import os
# import cv2
# from ultralytics import YOLO
#
# # path video
# video_path = os.path.join('cctv.mp4')
# link_camera1 = "rtsp://admin:BNNNRU@192.168.1.12:54/onvif1"
# cap = cv2.VideoCapture("rtsp://admin:BNNNRU@192.168.1.12:54/onvif1")
#
# # cap.set(cv2.CAP_PROP_FRAME_WIDTH, 160)
# # cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 120)
#
# # cap.set(3,200)
# # cap.set(4,50)
#
# # load model
# model = YOLO("yolov8n.pt")
# # results = model(frame, show=True, stream=True) # list of Results obj ,return_outputs=True
#
# ret = True
# #Loop open camera
# while ret:
#     #Read frame
#     ret, frame = cap.read()
#
#     if ret:
#         #track obj
#         results = model.track(frame)
#
#         annotated_frame = results[0].plot()
#
#         #visualize
#         cv2.imshow("YOLOV8", annotated_frame)
#
#         if cv2.waitKey(25) & 0xFF == ord('q'):
#             break
#
#         #export data
#         # for result in results:
#         #     boxes = result[0].boxes.numpy()
#         #     for box in boxes:
#         #         print("class", box.cls)
#         #         print("xyxy", box.xyxy)
#         #         print("conf", box.conf)
#
#     else:
#         break
# cap.release()
# cv2.destroyAllWindows()

# def show_frame():
#     cv2.imshow("show", frame)
#     cv2.waitKey(1)
#
# for result, frame in results:
#     show_frame()
#     boxes = result[0].boxes.numpy()
#     for box in boxes:
#         print("class", box.cls)
#         print("xyxy", box.xyxy)
#         print("conf", box.conf)

#while ret:
        # read frames

        # detect objects
        # track objects

        # plot results
        # cv2.rectangle
        # cv2.putText

        # visualize
       # cv2.imshow('frame', frame_)
       # if cv2.waitKey(25) & 0xFF == ord('q'):
       #     break









from ultralytics import YOLO
import cv2


# load yolov8 model
model = YOLO('yolov8n.pt')

# load video
video_path = 'cctv.mp4'
link_camera1 = "rtsp://admin:BNNNRU@192.168.1.12:554/onvif1"
cap = cv2.VideoCapture(0)

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalcatface.xml')
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

ret = True
# read frames
while ret:
    ret, frame = cap.read()
    #resize video
    frame = cv2.resize(frame,(854,480))

    #face detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 5)
        roi_gray = gray[y:y + w, x:x + w]
        roi_color = frame[y:y + h, x:x + w]
        eyes = eye_cascade.detectMultiScale(roi_gray, 1.3, 5)

        for (ex, ey, ew, eh) in eyes:
            cv2.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 5)
        break

    # tracking
    if ret:
        # detect objects

        # track objects
        results = model.track(frame, tracker="bytetrack.yaml", classes=0)


        # plot results
        # cv2.rectangle

        # cv2.putText
        annotated_frame = results[0].plot()

        # visualize
        cv2.imshow('YOLOV8', annotated_frame)

        if cv2.waitKey(25) & 0xFF == ord('q'):
            break

        #export data
        for result in results:
            boxes = result[0].boxes.numpy()
            for box in boxes:
                print("class", box.cls)
                print("xyxy", box.xyxy)
                print("conf", box.conf)

    else:
        break
cap.release()
cv2.destroyAllWindows()