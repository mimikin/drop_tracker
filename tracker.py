import cv2 as cv
import numpy as np

backSub = cv.createBackgroundSubtractorKNN()
vid = cv.VideoCapture('033121_5cm_6ml_yellow_elec.mov')

while vid.isOpened():
    ret, frame = vid.read()
    if frame is None:
        break


    blur = cv.GaussianBlur(frame, (19, 19), 0)
    hsv = cv.cvtColor(blur, cv.COLOR_BGR2HSV)
    # grey = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    # th = cv.adaptiveThreshold(grey, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C,cv.THRESH_BINARY, 5, 2)
    # # fgMask = backSub.apply(frame)
    # edges = cv.Canny(blur, 50, 150)


    lower_yellow = np.array([15, 110, 20])
    higher_yellow = np.array([60, 255, 255])

    mask = cv.inRange(hsv, lower_yellow, higher_yellow)
    res = cv.bitwise_and(frame, frame, mask=mask)
    res_grey = cv.cvtColor(cv.cvtColor(res, cv.COLOR_HSV2BGR), cv.COLOR_BGR2GRAY)
    res_th1 = cv.adaptiveThreshold(res_grey, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY, 11, 2)
    ret, res_th2 = cv.threshold(res_grey, 20, 255, cv.THRESH_BINARY)

    cv.imshow('Frame', frame)
    # cv.imshow('Threshhold', th)
    # cv.imshow('FG Mask', fgMask)
    # cv.imshow('Edges', edges)
    # cv.imshow('Color', res)
    # cv.imshow('Color', res_th1)
    # cv.imshow('Color_Grey', res_th2)

    contours, hierarchy = cv.findContours(res_th2, 1, 2)
    cnt = contours[0]
    M = cv.moments(cnt)
    cx = (M['m10'] / M['m00'])
    cy = (M['m01'] / M['m00'])
    print(cx, cy)

    keyboard = cv.waitKey(30)
    if keyboard == 'q' or keyboard == 27:
        break

vid.release()
cv.destroyAllWindows()



