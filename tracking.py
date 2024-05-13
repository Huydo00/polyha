from ultralytics import YOLO
import cv2


# load yolov8 model
model = YOLO('yolov8n.pt')

# load video
video_path = 'cctv.mp4'
link_camera1 = "rtsp://admin:BNNNRU@192.168.1.12:554/onvif1"
cap = cv2.VideoCapture(video_path)

ret = True
# read frames
while ret:
    ret, frame = cap.read()

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