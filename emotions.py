import cv2
import os
import multiprocessing
emo = ["angry","blink","blink2","happy","happy2","happy3","sad","sleep"]

image_folder = 'emotions/sleep'
images = [img for img in os.listdir(image_folder) if img.endswith(".png")]

images.sort()
cv2.namedWindow("Slideshow", cv2.WINDOW_NORMAL)

while True:
    for image in images:
        image_path = os.path.join(image_folder, image)
        frame = cv2.imread(image_path)
        cv2.imshow("Slideshow", frame)
        cv2.waitKey(100)

cv2.destroyAllWindows()