import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

img = cv.imread("C:\\mydoc\\ElectricDesign_project\\nobodyfly\\IT-CUP_JSBandCV\\test_pic\\42.jpg", cv.IMREAD_COLOR)

"""二值化"""
"""
全局阈值化处理
src：表示的是图片源
thresh：表示的是阈值（分割值）
maxval：表示的是最大值
type：表示的是这里划分的时候使用的是什么类型的算法，包含以下几种
"""
def image_binarization(img):
    # 将图片转为灰度图
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    retval, dst = cv.threshold(gray, 127, 255, cv.THRESH_BINARY)
    # 最大类间方差法(大津算法)，thresh会被忽略，自动计算一个阈值
    # retval, dst = cv.threshold(gray, 127, 255, cv.THRESH_BINARY | cv.THRESH_OTSU)
    cv.imshow('binary.jpg', dst)


"""
局部阈值化处理
src：需要进行二值化的一张灰度图像
maxValue：满足条件的像素点需要设置的灰度值。（将要设置的灰度值）
adaptiveMethod：自适应阈值算法。可选ADAPTIVE_THRESH_MEAN_C 或 ADAPTIVE_THRESH_GAUSSIAN_C
thresholdType：opencv提供的二值化方法，只能THRESH_BINARY或者THRESH_BINARY_INV
blockSize：要分成的区域大小，上面的N值，取奇数
C：常数，每个区域计算出的阈值的基础上在减去这个常数作为这个区域的最终阈值，可以为负数
dst：输出图像，可以忽略
"""
def image_binarization_part_situation(img):
    # 转灰度图
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    # 图像压缩(非必要步骤)
    # new_gray = np.uint8((255 * (gray/255.0)**1.4))
    new_gray = gray
    # 二值化
    dst = cv.adaptiveThreshold(new_gray, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY, 15, 1)
    # 中值滤波
    dwt = cv.medianBlur(dst, 5)
    cv.imshow("bin_part", dst)


if __name__ == "__main__":
    cv.imshow("src_image", img)
    # image_binarization(img)
    image_binarization_part_situation(img)
    cv.waitKey(0)
    cv.destroyAllWindows()