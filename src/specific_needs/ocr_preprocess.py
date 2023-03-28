import cv2 as cv
import numpy as np


# ocr的前置处理
def resize(image, width=None, height=None, inter=cv.INTER_AREA):
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
    resized = cv.resize(image, dim, interpolation=inter)
    return resized


# 排序四个点
def order_points(pts):
    # 一共4个坐标点
    rect = np.zeros((4, 2), dtype="float32")
    # 按顺序找到对应坐标0123分别是 左上，右上，右下，左下
    # 计算左上，右下
    s = pts.sum(axis = 1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    # 计算右上和左下
    diff = np.diff(pts, axis = 1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect


# 实现透视变换
def four_point_transform(image, pts):
    # 获取输入坐标点
    rect = order_points(pts)
    (tl, tr, br, bl) = rect
    # 计算输入的w和h值
    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))
    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = max(int(heightA), int(heightB))
    # 变换后对应坐标位置（-1只是为了防止有误差出现，不-1也可以。）
    dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]], dtype = "float32")
    # 计算变换矩阵
    M = cv.getPerspectiveTransform(rect, dst)
    warped = cv.warpPerspective(image, M, (maxWidth, maxHeight))
    # 返回变换后结果
    return warped


def ocr_preprocess():
    # 读取输入
    img = cv.imread('test.jpg')
    # 坐标也会相同变化
    ratio = img.shape[0] / 500.0
    orig = img.copy()
    image = resize(orig, height=500)

    # 预处理
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    gray = cv.GaussianBlur(gray, (5, 5), 0)  # 高斯滤波
    edged = cv.Canny(gray, 75, 200)  # Canny边缘检测
    # 展示预处理结果
    print("STEP 1: 边缘检测")
    cv.imshow("Image", image)
    cv.imshow("Edged", edged)

    # 轮廓检测，最大的为所需要的轮廓
    cnts = cv.findContours(edged.copy(), cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE)[0]
    cnts = sorted(cnts, key=cv.contourArea, reverse=True)[:5]

    # 遍历轮廓
    for c in cnts:
        # 计算轮廓近似
        peri = cv.arcLength(c, True)
        # c表示输入的点集
        # epsilon表示从原始轮廓到近似轮廓的最大距离，它是一个准确度参数
        # True表示封闭的
        approx = cv.approxPolyDP(c, 0.02 * peri, True)
        # 4个点的时候就拿出来
        if len(approx) == 4:
            screenCnt = approx
            break
    # 展示结果
    print("STEP 2: 获取轮廓")
    cv.drawContours(image, [screenCnt], -1, (0, 255, 0), 2)
    cv.imshow("Outline", image)
    # 透视变换
    warped = four_point_transform(orig, screenCnt.reshape(4, 2) * ratio)
    # 二值处理
    warped = cv.cvtColor(warped, cv.COLOR_BGR2GRAY)
    ref = cv.threshold(warped, 100, 255, cv.THRESH_BINARY)[1]
    cv.imwrite('scan.jpg', ref)
    # 展示结果
    print("STEP 3: 变换")
    cv.imshow("Scanned", ref)


if __name__ == "__main__":
    ocr_preprocess();
    cv.waitKey(0)
    cv.destroyAllWindows()