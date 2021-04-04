import numpy as np
import cv2 as cv

vid = cv.VideoCapture('033121_5cm_6ml_yellow_elec.mov')

while vid.isOpened():
    ret, frame = vid.read()
    # if frame is read correctly ret is True
    if not ret:
        print("Can't receive frame (stream end?). Exiting ...")
        break
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    cv.imshow('frame', gray)
    if cv.waitKey(1) == ord('q'):
        break
vid.release()
cv.destroyAllWindows()