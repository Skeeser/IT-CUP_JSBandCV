import numpy as np
import os
import cv2
import math
import pytesseract
from PIL import Image
from cv2 import namedWindow
from scipy import misc, ndimage


# ocr的前置处理
def resize(image, width=None, height=None, inter=cv2.INTER_AREA):
    dim = None
    (h, w) = image.shape[:2]
    if width is None and height is None:
        return image
    if width is None:
        r = height / float(h)
        dim = (int(w * r), height)
    else:
        r = width / float(w)
        dim = (width, int(h * r))
    resized = cv2.resize(image, dim, interpolation=inter)
    return resized


def color_equilibrium(image):
    (b, g, r) = cv2.split(image)  # 通道分解
    bH = cv2.equalizeHist(b)
    gH = cv2.equalizeHist(g)
    rH = cv2.equalizeHist(r)
    result = cv2.merge((bH, gH, rH), )  # 通道合成
    return result


def rotate(image, angle, center=None, scale=1.0):
    (w, h) = image.shape[0:2]
    if center is None:
        center = (w//2, h//2)
    wrapMat = cv2.getRotationMatrix2D(center, angle, scale)
    return cv2.warpAffine(image, wrapMat, (h, w))


def blackfilter(image):
    (b, g, r) = cv2.split(image)  # 通道分解
    bH = cv2.equalizeHist(b)
    gH = cv2.equalizeHist(g)
    rH = cv2.equalizeHist(r)
    image = cv2.merge((bH, gH, rH), )  # 通道合成

    img_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # 得到目标颜色的二值图像，用作cv2.bitwise_and()的掩模
    img_target = cv2.inRange(img_hsv, (0, 0, 0), (100, 100, 100))
    # img_specifiedColor = cv2.bitwise_or(img_hsv, img_hsv, mask=img_target)
    img_specifiedColor = cv2.bitwise_not(img_target)
    return img_specifiedColor


def image_binarization_part_situation(img):
    # 转灰度图
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # 图像压缩(非必要步骤)
    # new_gray = np.uint8((255 * (gray/255.0)**1.4))
    new_gray = gray
    # 二值化
    dst = cv2.adaptiveThreshold(new_gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 15, 1)

    # 中值滤波
    # dwt = cv2.medianBlur(dst, 5)
    return dst

def ocrprocess(image):
    preprocess = 'blur'  # thresh

    image = cv2.imread('scan.jpg')
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    if preprocess == "thresh":
        gray = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]

    if preprocess == "blur":
        gray = cv2.medianBlur(gray, 3)

    filename = "{}.png".format(os.getpid())
    cv2.imwrite(filename, gray)

    text = pytesseract.image_to_string(Image.open(filename))
    print(text)
    os.remove(filename)


# 使用霍夫变换
def getCorrect2():
    # 读取图片，灰度化
    src = cv2.imread("D:\\AllMyProject\\ElectricDesign_project\\nobodyfly\\IT-CUP_JSBandCV\\test_pic\\ocr_test_pic2.jpg")
    # src = color_equilibrium(src)
    # 坐标也会相同变化
    ratio = src.shape[0] / 500.0
    orig = src.copy()
    src = resize(orig, height=1000)

    # showAndWaitKey("src", src)
    gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
    # showAndWaitKey("gray", gray)
    # 腐蚀、膨胀
    kernel = np.ones((5, 5), np.uint8)
    erode_Img = cv2.erode(gray, kernel)
    eroDil = cv2.dilate(erode_Img, kernel)
    # showAndWaitKey("eroDil", eroDil)
    # 边缘检测
    canny = cv2.Canny(eroDil,50,150)
    # showAndWaitKey("canny",canny)
    # 霍夫变换得到线条
    lines = cv2.HoughLinesP(canny, 0.8, np.pi / 180, 90,minLineLength=100,maxLineGap=10)
    drawing = np.zeros(src.shape[:], dtype=np.uint8)
    # 画出线条
    for line in lines:
        x1, y1, x2, y2 = line[0]
        cv2.line(drawing, (x1, y1), (x2, y2), (0, 255, 0), 1, lineType=cv2.LINE_AA)
    
    # showAndWaitKey("houghP",drawing)
    """
    计算角度,因为x轴向右，y轴向下，所有计算的斜率是常规下斜率的相反数，我们就用这个斜率（旋转角度）进行旋转
    """
    k = float(y1-y2)/(x1-x2)
    thera = np.degrees(math.atan(k))
    print(thera)

    """
    旋转角度大于0，则逆时针旋转，否则顺时针旋转
    """
    rotateImg = rotate(src, thera)
    showAndWaitKey("rotateImg", rotateImg)
    # blackimg = blackfilter(rotateImg)
    # blackimg = image_binarization_part_situation(rotateImg)

    warped = cv2.cvtColor(rotateImg, cv2.COLOR_BGR2GRAY)
    ref = cv2.threshold(warped, 150, 255, cv2.THRESH_BINARY)[1]
    showAndWaitKey("bImg", ref)
    ocrprocess(ref)
    cv2.destroyAllWindows()
    # cv2.imwrite('result.jpg', rotateImg)


def showAndWaitKey(winName, img):
    cv2.imshow(winName, img)
    cv2.waitKey()


if __name__ == "__main__":              
    getCorrect2()
