import cv2 as cv
import numpy as np
import pandas as pd


def frame_cutter(frame, type, position):
    if type == "center":
        cutted_frame = frame[position[0][0]:position[0][1], position[1][0]:position[1][1]]
    return cutted_frame


def centroid_calculation(processed_frame):
    m = cv.moments(processed_frame)  # calculates the moments. see en.wikipedia.org/wiki/Image_moment
    cw = (m['m10'] / m['m00'])  # calculates the centroid
    ch = (m['m01'] / m['m00'])
    return cw, ch


def csv_output(t, x, y, file_path):
    pd.DataFrame(t, x, y).to_csv(file_path)  # output of the csv file


def drop_filter(frame, color):
    kernel = np.ones((7, 7), np.uint8)
    blur = cv.GaussianBlur(frame, (25, 25), 0)
    hsv = cv.cvtColor(blur, cv.COLOR_BGR2HSV)

    lower_yellow = np.array([5, 80, 20])
    higher_yellow = np.array([80, 255, 255])

    if color == "yellow":
        mask = cv.inRange(hsv, lower_yellow, higher_yellow)

    res = cv.bitwise_and(frame, frame, mask=mask)

    res_grey = cv.cvtColor(cv.cvtColor(res, cv.COLOR_HSV2BGR), cv.COLOR_BGR2GRAY)

    ret, res_th = cv.threshold(res_grey, 20, 255, cv.THRESH_BINARY)

    filtered_frame = cv.morphologyEx(res_th, cv.MORPH_CLOSE, kernel, iterations=2)

    return filtered_frame, res_th


def trail_display(frame, w, h):
    trails = frame
    if len(w) > 30:
        for i in range(-1, -25, -3):
            trails[int(h[i]), int(w[i])] = [0, 0, 255]
            trails[int(h[i]) + 1, int(w[i])] = [0, 0, 255]
            trails[int(h[i]), int(w[i]) + 1] = [0, 0, 255]
            trails[int(h[i]) - 1, int(w[i])] = [0, 0, 255]
            trails[int(h[i]), int(w[i]) - 1] = [0, 0, 255]
    cv.imshow('Trails', trails)
    cv.waitKey(100)

    return


def drop_tracker(video_path, fps, position, start=1, end=-1, color="yellow", d2pixel=1, file_path="time-series.csv"):
    vid = cv.VideoCapture(video_path)  # open the video from input path
    length = int(vid.get(cv.CAP_PROP_FRAME_COUNT))  # find the length of the video
    vid.set(cv.CAP_PROP_POS_FRAMES, start - 1)

    frame_count = 0
    w = []  # lists for centroid storage
    h = []
    t = []

    if end == -1:
        end = length - 1  # assign of the end frame for analysis

    while vid.isOpened():
        ret, frame = vid.read()
        frame_count = frame_count + 1

        if frame_count == 1:
            height, width = frame.shape[:2]

        if frame_count > end - start:
            break

        # print(frame_count)

        cutted_frame = frame_cutter(frame, "center", position)  # cutting down the frame

        # print(frame_count)

        filtered_frame, res_th = drop_filter(cutted_frame,
                                             color)  # get the black-and-white frame with only drop as white

        cw, ch = centroid_calculation(filtered_frame)

        w.append(cw)
        h.append(ch)
        t.append(frame_count / fps)

        if (frame_count) % 10 == 0:  # output frame-processing progress in terms of number of frames processed
            print(frame_count)

        trail_display(cutted_frame, w, h)
        cv.imshow("grey", filtered_frame)

    x = (np.array(w) + position[1][1] - height // 2) * d2pixel
    y = (np.array(h) + position[0][1] - width // 2) * d2pixel
    t = np.array(t)

    csv_output(t, x, y, file_path)

    vid.release()
    cv.destroyAllWindows()


def main():
    position = [[720, 1440], [1280, 2560]]
    drop_tracker("033121_5cm_6ml_225C_yellow_elec.mp4", 60, position, start=9000, end=18000, color="yellow",
                 d2pixel=0.0652647, file_path="mass5.csv")
    # this is the place for testing


if __name__ == "__main__":
    main()
