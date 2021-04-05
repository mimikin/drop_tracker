import cv2 as cv
import numpy as np
import pandas as pd


def coloriden_th_frame(frame, color):
    blur = cv.GaussianBlur(frame, (19, 19), 0)
    hsv = cv.cvtColor(blur, cv.COLOR_BGR2HSV)

    lower_yellow = np.array([15, 110, 20])
    higher_yellow = np.array([60, 255, 255])

    if color == "yellow":
        mask = cv.inRange(hsv, lower_yellow, higher_yellow)

    res = cv.bitwise_and(frame, frame, mask=mask)
    res_grey = cv.cvtColor(cv.cvtColor(res, cv.COLOR_HSV2BGR), cv.COLOR_BGR2GRAY)
    ret, res_th2 = cv.threshold(res_grey, 20, 255, cv.THRESH_BINARY)
    kernel = np.ones((11, 11), np.uint8)
    dilation = cv.dilate(res_th2, kernel, iterations=1)

    return dilation

def show_trails(frame, inw, inh):
    trails = frame
    if len(inw) > 30:
        for i in range(-1, -30, -3):
            trails[inh[i], inw[i]] = [0, 0, 255]
            trails[inh[i] + 1, inw[i]] = [0, 0, 255]
            trails[inh[i], inw[i] + 1] = [0, 0, 255]
            trails[inh[i] - 1, inw[i]] = [0, 0, 255]
            trails[inh[i], inw[i] - 1] = [0, 0, 255]
    cv.imshow('Trails', trails)
    keyboard = cv.waitKey(5)
    return

# video_path: local path for video file
# frame_size: overall size of the original frame
# frame_start: where should the top-left corner of the trimmed frame be located on the original video
# start: which frame (starting from 1) to start porcess
# end: which frame to end process
# color: currently only yellow available
# d2pixel: actual distance to pixel length, default set to 1
# display: display the trails with a open video window if set to True

def drop_tracker_csv(video_path, fps, frame_size_w, frame_size_h, frame_start_w, frame_start_h, start=1,
                     end=100000000000000, color="yellow", d2pixel=1, display=False):

    vid = cv.VideoCapture(video_path)  # open the video from input path
    length = int(vid.get(cv.CAP_PROP_FRAME_COUNT))  # find the length of the video
    count = 0

    if end == 100000000000000:
        end = length - 1  # assign of the end frame for analysis

    w = []  # lists for centroid storage
    h = []
    t = []
    inw = []  # list for interger values of centroid, only for trails display
    inh = []

    while vid.isOpened():
        ret, frame = vid.read()
        frame = frame[frame_start_h:frame_size_h - frame_start_h, frame_start_w:frame_size_w - frame_start_w]
        # cutting down the frame
        count = count + 1

        if count < start:
            continue
        if count == end:
            break

        processed_frame = coloriden_th_frame(frame, color) # get the black-and-white frame

        m = cv.moments(processed_frame) # calculates the moments. see en.wikipedia.org/wiki/Image_moment
        # print(M)
        cw = (m['m10'] / m['m00']) # calculates the centroid
        ch = (m['m01'] / m['m00'])
        inch = int(m['m01'] / m['m00']) # calculates the centroid trimmed to intergers for displaying trails
        incw = int(m['m10'] / m['m00'])

        # print(cw, ch)
        w.append(cw)
        h.append(ch)
        t.append((count - 1) * fps)

        inw.append(incw)
        inh.append(inch)

        if display:
            show_trails(frame, inw, inh) # show trails on displayed frame

        # cv.imshow('Frame', frame)
        # cv.imshow('Trails', trails)
        # cv.imshow('Threshhold', th)
        # cv.imshow('FG Mask', fgMask)
        # cv.imshow('Edges', edges)
        # cv.imshow('Color', res)
        # cv.imshow('Color', res_grey)
        # cv.imshow('Color_Grey', res_th2)
        # cv.imshow('Color', dilation)

        if (count % 100 == 0): # output frame-processing progress in terms of number of frames processed
            print(count)

    x = (np.array(w) + frame_start_w) * d2pixel
    y = (np.array(h) + frame_start_h) * d2pixel
    t = np.array(t)
    pd.DataFrame(t, x, y).to_csv(video_path + "_time-series.csv") # output of the csv file

    vid.release()
    cv.destroyAllWindows()


def main():
    drop_tracker_csv("033121_0cm_4ml_225C_yellow_elec_edited.mp4", 60, 600, 300, 1920, 1080, display = True)
    # this is the place for testing


if __name__ == "__main__":
    main()
