import cv2 as cv
import numpy as np

img = cv.imread("C:\\mydoc\\ElectricDesign_project\\nobodyfly\\IT-CUP_JSBandCV\\test_pic\\42.jpg", cv.IMREAD_COLOR)


# 定义滑动条回调函数，此处pass用作占位语句保持程序结构的完整性
def nothing(x):
    pass


def hsv_color_identify(image):
    # 定义窗口名称
    winName = 'Colors of the rainbow'
    img_original = image
    # 颜色空间的转换
    img_hsv = cv.cvtColor(img_original, cv.COLOR_BGR2HSV)
    # 新建窗口
    cv.namedWindow(winName)
    # 新建6个滑动条，表示颜色范围的上下边界，这里滑动条的初始化位置即为黄色的颜色范围
    cv.createTrackbar('LowerbH', winName, 0, 255, nothing)
    cv.createTrackbar('LowerbS', winName, 0, 255, nothing)
    cv.createTrackbar('LowerbV', winName, 0, 255, nothing)
    cv.createTrackbar('UpperbH', winName, 255, 255, nothing)
    cv.createTrackbar('UpperbS', winName, 255, 255, nothing)
    cv.createTrackbar('UpperbV', winName, 255, 255, nothing)
    while (1):
        # 函数cv2.getTrackbarPos()范围当前滑块对应的值
        lowerbH = cv.getTrackbarPos('LowerbH', winName)
        lowerbS = cv.getTrackbarPos('LowerbS', winName)
        lowerbV = cv.getTrackbarPos('LowerbV', winName)
        upperbH = cv.getTrackbarPos('UpperbH', winName)
        upperbS = cv.getTrackbarPos('UpperbS', winName)
        upperbV = cv.getTrackbarPos('UpperbV', winName)
        # 得到目标颜色的二值图像，用作cv2.bitwise_and()的掩模
        img_target = cv.inRange(img_original, (lowerbH, lowerbS, lowerbV), (upperbH, upperbS, upperbV))
        # 输入图像与输入图像在掩模条件下按位与，得到掩模范围内的原图像
        img_specifiedColor = cv.bitwise_and(img_original, img_original, mask=img_target)
        cv.imshow(winName, img_specifiedColor)


if __name__ == "__main__":
    cv.imshow("src_image", img)
    hsv_color_idenfic(img)
    cv.waitKey(0)
    cv.destroyAllWindows()