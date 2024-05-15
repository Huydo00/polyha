from ultralytics import YOLO
import cv2


# load yolov8 model
model = YOLO('yolov8n.pt')

# load video
video_path = 'cctv.mp4'
link_camera1 = "rtsp://admin:BNNNRU@192.168.1.12:554/onvif1"
cap = cv2.VideoCapture(1)


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