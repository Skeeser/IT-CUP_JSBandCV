import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt


def nothing(x):  # 滑动条的回调函数
    pass


src = cv.imread('C:\\mydoc\\ElectricDesign_project\\nobodyfly\\IT-CUP_JSBandCV\\test_pic\\42.jpg')
srcBlur = cv.GaussianBlur(src, (3, 3), 0)
gray = cv.cvtColor(srcBlur, cv.COLOR_BGR2GRAY)
edges = cv.Canny(gray, 50, 150, apertureSize=3)
WindowName = 'Approx'  # 窗口名
cv.namedWindow(WindowName, cv.WINDOW_AUTOSIZE)  # 建立空窗口

cv.createTrackbar('threshold', WindowName, 0, 60, nothing)  # 创建滑动条
cv.createTrackbar('minLineLength', WindowName, 0, 100, nothing)  # 创建滑动条
cv.createTrackbar('maxLineGap', WindowName, 0, 100, nothing)  # 创建滑动条


def demo1():
    while(1):
        img = src.copy()
        threshold = 100 + 2 * cv.getTrackbarPos('threshold', WindowName)  # 获取滑动条值

        lines = cv.HoughLines(edges, 1, np.pi / 180, threshold)

        for line in lines:
            rho = line[0][0]
            theta = line[0][1]
            a = np.cos(theta)
            b = np.sin(theta)
            x0 = a * rho
            y0 = b * rho
            x1 = int(x0 + 1000 * (-b))
            y1 = int(y0 + 1000 * (a))
            x2 = int(x0 - 1000 * (-b))
            y2 = int(y0 - 1000 * (a))

            cv.line(img, (x1, y1), (x2, y2), (0, 0, 255), 2)

        cv.imshow(WindowName, img)
        k = cv.waitKey(1) & 0xFF
        if k == 27:
            break


def demo2():
    while (1):
        img = src.copy()
        threshold = cv.getTrackbarPos('threshold', WindowName)  # 获取滑动条值
        minLineLength = 2 * cv.getTrackbarPos('minLineLength', WindowName)  # 获取滑动条值
        maxLineGap = cv.getTrackbarPos('maxLineGap', WindowName)  # 获取滑动条值

        lines = cv.HoughLinesP(edges, 1, np.pi / 180, threshold, minLineLength, maxLineGap)

        for line in lines:
            x1 = line[0][0]
            y1 = line[0][1]
            x2 = line[0][2]
            y2 = line[0][3]
            cv.line(img, (x1, y1), (x2, y2), (0, 255, 0), 2)

        cv.imshow(WindowName, img)
        k = cv.waitKey(1) & 0xFF
        if k == 27:
            break


if __name__ == "__main__":
    cv.imshow("src_image", src)
    # demo1()
    demo2()
    cv.waitKey(0)
    cv.destroyAllWindows()