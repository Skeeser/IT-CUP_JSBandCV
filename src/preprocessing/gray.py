import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

img = cv.imread("C:\\mydoc\\ElectricDesign_project\\nobodyfly\\IT-CUP_JSBandCV\\test_pic\\42.jpg", cv.IMREAD_COLOR)  # 读取指定位置的彩色图像

"""
cv灰度化库函数
"""
def set_gray(image):
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    cv.imshow("img", gray)  # 显示图片



"""
cv灰度化分量法
"""
def set_gray_RGB(image):
    height, width, channel = image.shape

    # 创建空数组
    R = np.zeros(image.shape, np.uint8)
    G = np.zeros(image.shape, np.uint8)
    B = np.zeros(image.shape, np.uint8)

    for i in range(height):
        for j in range(width):
            R[i, j] = image[i, j, 0]
            G[i, j] = image[i, j, 1]
            B[i, j] = image[i, j, 2]

    cv.imshow('R', R)
    # cv.imshow('G', G)
    # cv.imshow('B', B)



if __name__ == "__main__":
    set_gray(img)
    cv.waitKey(0)  # 等待键盘触发事件，释放窗口




