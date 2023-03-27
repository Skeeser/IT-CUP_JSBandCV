import cv2 as cv
import numpy as np

img = cv.imread("C:\\mydoc\\ElectricDesign_project\\nobodyfly\\IT-CUP_JSBandCV\\test_pic\\42.jpg", cv.IMREAD_COLOR)
# img_gray = cv.imread("C:\\mydoc\\ElectricDesign_project\\nobodyfly\\IT-CUP_JSBandCV\\test_pic\\42.jpg", 0)

"""
直方图均衡
增加图像的对比度
"""

# 直方图均衡
def set_equilibrium(image):
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    equ = cv.equalizeHist(gray)
    cv.imshow('img', equ)


# CLAHE 有限对比适应性直方图均衡化
def clahe_equilibrium(image):
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    clahe = cv.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    cl1 = clahe.apply(gray)
    cv.imshow('clahe_2.jpg', cl1)


def color_equilibrium(image):
    (b, g, r) = cv.split(image)  # 通道分解
    bH = cv.equalizeHist(b)
    gH = cv.equalizeHist(g)
    rH = cv.equalizeHist(r)
    result = cv.merge((bH, gH, rH), )  # 通道合成
    cv.imshow('dst', result)


if __name__ == "__main__":
    cv.imshow("src_image", img)
    # set_equilibrium(img)
    # clahe_equilibrium(img)
    color_equilibrium(img)
    cv.waitKey(0)
    cv.destroyAllWindows()