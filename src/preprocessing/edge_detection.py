import cv2 as cv
import numpy as np

img = cv.imread("C:\\mydoc\\ElectricDesign_project\\nobodyfly\\IT-CUP_JSBandCV\\test_pic\\42.jpg", cv.IMREAD_COLOR)
# img_gray = cv.imread("C:\\mydoc\\ElectricDesign_project\\nobodyfly\\IT-CUP_JSBandCV\\test_pic\\42.jpg", 0)


"""
canny = cv2.Canny(image, threshold1, threshold2[, edges[, apertureSize[, L2gradient ]]]) 
参数：
第一个参数 处理的原图像，该图像必须为单通道的灰度图；
第二个参数 最小阈值；
第三个参数 最大阈值。
"""
# Canny边缘检测--最优
def set_canny(image):
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    blur = cv.GaussianBlur(gray, (3, 3), 0)  # 用高斯滤波处理原图像降噪
    canny = cv.Canny(blur, 50, 150)  # 50是最小阈值,150是最大阈值
    cv.imshow('canny', canny)
    # 反色
    nocanny = cv.bitwise_not(canny)
    cv.imshow('nocanny', nocanny)


"""
Sobel_x_or_y = cv2.Sobel(src, ddepth, dx, dy, dst, ksize, scale, delta, borderType)
参数：
第一个参数是需要处理的图像；
第二个参数是图像的深度，-1表示采用的是与原图像相同的深度。目标图像的深度必须大于等于原图像的深度；
dx和dy表示的是求导的阶数，0表示这个方向上没有求导，一般为0、1、2。
"""
# sobel算子
def set_sobel(image):
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    x = cv.Sobel(gray, cv.CV_16S, 1, 0)
    y = cv.Sobel(gray, cv.CV_16S, 0, 1)
    # cv2.convertScaleAbs(src[, dst[, alpha[, beta]]])
    # 可选参数alpha是伸缩系数，beta是加到结果上的一个值，结果返回uint类型的图像
    Scale_absX = cv.convertScaleAbs(x)  # convert 转换  scale 缩放
    Scale_absY = cv.convertScaleAbs(y)
    result = cv.addWeighted(Scale_absX, 0.5, Scale_absY, 0.5, 0)
    cv.imshow('result', result)
    # 反色
    noresult = cv.bitwise_not(result)
    cv.imshow('noresult', noresult)


def edge_demo(image):
    blurred = cv.GaussianBlur(image, (3, 3), 0)  # 高斯模糊
    gray = cv.cvtColor(blurred,cv.COLOR_BGR2GRAY)  # 灰路图像
    xgrad = cv.Sobel(gray, cv.CV_16SC1, 1, 0)  # xGrodient
    ygrad = cv.Sobel(gray, cv.CV_16SC1, 0, 1)  # yGrodient
    edge_output = cv.Canny(xgrad, ygrad, 100, 150)  # edge
    return edge_output


# 边缘检测和轮廓调节
def contours_demo(image):
    blurred = cv.GaussianBlur(image, (3, 3), 0)
    gray = cv.cvtColor(blurred, cv.COLOR_RGB2GRAY)  # 灰路图像
    ret, binary = cv.threshold(gray, 0, 255,cv.THRESH_BINARY| cv.THRESH_OTSU) # 图像二值化
    ret, binary = cv.threshold(gray, 68, 255, cv.THRESH_BINARY )  # 图像二值化
    cv.imshow("binary image", binary)
    binary = edge_demo(image)
    contours, heriachy = cv.findContours(binary, cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE)
    for i, contour in enumerate(contours):
        cv.drawContours(image, contours,i,(0,0,255),6)   #6的改为-1可以填充
        print(i)
    cv.imshow("detect contours", image)


if __name__ == "__main__":
    cv.imshow("src_image", img)
    # set_canny(img)
    # set_sobel(img)
    contours_demo(img)
    cv.waitKey(0)
    cv.destroyAllWindows()