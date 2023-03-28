import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

img = cv.imread("C:\\mydoc\\ElectricDesign_project\\nobodyfly\\IT-CUP_JSBandCV\\test_pic\\42.jpg", cv.IMREAD_COLOR)
# img_gray = cv.imread("C:\\mydoc\\ElectricDesign_project\\nobodyfly\\IT-CUP_JSBandCV\\test_pic\\42.jpg", 0)


"""轮廓检测"""
def profile_detection(image):
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    ret, binary = cv.threshold(gray, 127, 255, cv.THRESH_BINARY)
    contours, hierarchy = cv.findContours(binary, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    cv.drawContours(image, contours, -1, (0, 0, 255), 3)
    cv.imshow("img", image)


"""
霍夫圆----检测圆形
circle = cv2.HoughCircles(image, method, dp, minDist, param1=100, param2=100, minRadius=0, maxRadius=0)
参数：
image：输入图像（灰度图）
method：使用霍夫变换圆检测的算法，参数为cv2.HOUGH_GRADIENT
dp：霍夫空间的分辨率，dp = 1
时表示霍夫空间与输入图像空间的大小一致，dp = 2
时霍夫空间是输入图像空间的一半，以此类推
minDist：为圆心之间的最小距离，如果检测到的两个圆心之间距离小于该值，则认为它们是同一个圆心
param1：边缘检测时使用Canny算子的高阈值，低阈值是高阈值的一半
param2∶检测圆心和确定半径时所共有的阈值
minRadius和maxRadius：为所检测到的圆半径的最小值和最大值
返回值：
circles：输出圆向量，包括三个浮点型的元素――圆心横坐标，圆心纵坐标和圆半径。
"""
def Hough_circle(image):
    # 进行中值滤波
    # dst = cv.medianBlur(image, 7)
    # 霍夫圆检测对噪声敏感，边缘检测消噪
    dst = cv.pyrMeanShiftFiltering(image, 10, 100)  # 边缘保留滤波EPF
    gray = cv.cvtColor(dst, cv.COLOR_BGR2GRAY)
    circles = cv.HoughCircles(gray, cv.HOUGH_GRADIENT, 1, 20, param1=150, param2=50, minRadius=0, maxRadius=0)
    circles = np.uint16(np.around(circles))  #把circles包含的圆心和半径的值变成整数
    for i in circles[0, :]:
        cv.circle(image, (i[0], i[1]), i[2], (0, 0, 255), 2)
        cv.circle(image, (i[0], i[1]), 2, (255, 0, 0), 2)
    cv.imshow("circle image", image)


"""
透视变换，详见ocr_preprocess.py
"""

if __name__ == "__main__":
    cv.imshow("src_image", img)
    # profile_detection(img)
    # Hough_circle(img)
    cv.waitKey(0)
    cv.destroyAllWindows()