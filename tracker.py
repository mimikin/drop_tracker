import cv2 as cv
import numpy as np
import pandas as pd


def com_tracker(video_path):
    count = int(0)
    vid = cv.VideoCapture(video_path)
    # vid = cv.VideoCapture("033121_0cm_4ml_225C_yellow_elec_edited.mp4")
    w = []
    h = []

    while vid.isOpened():
        ret, frame = vid.read()
        frame = frame[300:780, 600:1320]
        count = count + int(1)
        if ((frame is None) or (count == 12300)):
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
        # res_th1 = cv.adaptiveThreshold(res_grey, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY, 11, 2)
        ret, res_th2 = cv.threshold(res_grey, 20, 255, cv.THRESH_BINARY)
        kernel = np.ones((9, 9), np.uint8)
        dilation = cv.dilate(res_th2, kernel, iterations=1)

        # contours, hierarchy = cv.findContours(res_th2, 1, 2)
        # cnt = contours[0]
        M = cv.moments(dilation)
        # print(M)
        cw = int(M['m10'] / M['m00'])
        ch = int(M['m01'] / M['m00'])
        # print(cw, ch)

        w.append(cw)
        h.append(ch)

        trails = frame
        if len(w) > 30:
            for i in range(-1, -30, -3):
                trails[h[i], w[i]] = [0, 0, 255]
                trails[h[i] + 1, w[i]] = [0, 0, 255]
                trails[h[i], w[i] + 1] = [0, 0, 255]
                trails[h[i] - 1, w[i]] = [0, 0, 255]
                trails[h[i], w[i] - 1] = [0, 0, 255]

        # cv.imshow('Frame', frame)
        cv.imshow('Trails', trails)
        # cv.imshow('Threshhold', th)
        # cv.imshow('FG Mask', fgMask)
        # cv.imshow('Edges', edges)
        # cv.imshow('Color', res)
        # cv.imshow('Color', res_grey)
        # cv.imshow('Color_Grey', res_th2)
        # cv.imshow('Color', dilation)

        keyboard = cv.waitKey(5)
        # if keyboard == 'q' or keyboard == 27:
        #     break

        if (count % 100 == 0):
            print(count)

    width = (np.array(w)+600)*0.130325112
    height = (np.array(h)+300)*0.130325112
    pd.DataFrame(width, height).to_csv(video_path + "_time-series.csv")


    vid.release()
    cv.destroyAllWindows()


com_tracker("033121_0cm_4ml_225C_yellow_elec_edited.mp4")
