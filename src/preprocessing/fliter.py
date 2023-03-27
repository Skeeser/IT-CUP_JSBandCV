import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

img = cv.imread("C:\\mydoc\\ElectricDesign_project\\nobodyfly\\IT-CUP_JSBandCV\\test_pic\\42.jpg", cv.IMREAD_COLOR)

"""-------------------平滑------------------------"""
"""
均值模糊 : 去随机噪声有很好的去噪效果(低通滤波)
（1, 15）是垂直方向模糊，（15， 1）是水平方向模糊
"""
def set_blur(image):
    dst = cv.blur(image, (1, 15))
    cv.imshow("avg_blur_demo", dst)


"""
中值模糊  对椒盐噪声有很好的去燥效果（高通滤波）
"""
def set_median_blur(image):
    dst = cv.medianBlur(image, 5)
    cv.imshow("median_blur_demo", dst)


"""
高斯滤波是一种线性平滑滤波，适用于消除高斯噪声，广泛应用于图像处理的减噪过程。
通俗的讲，高斯滤波就是对整幅图像进行加权平均的过程，每一个像素点的值，都由其本身和邻域内的其他像素值经过加权平均后得到。
cv2.GaussianBlur(src, ksize, sigmaX, sigmaY, borderType)
src：输入的图像
ksize：高斯卷积核的大小。注意：卷积核的高度和宽度都应为奇数，且可以不同。
sigmaX：水平方向的标准差
sigmaY：垂直方向的标准差，默认值为0，表示与sigmaX相同
borderType：边界类型
"""
def set_gaussian_blur(image):
    img_gaussianBlur = cv.GaussianBlur(img, (3, 3), 1)
    cv.imshow("img_gaussianBlur", img_gaussianBlur)


"""
用户自定义模糊
下面除以25是防止数值溢出
"""
def set_custom_blur(image):
    kernel = np.ones([5, 5], np.float32)/25
    dst = cv.filter2D(image, -1, kernel)
    cv.imshow("custom_blur_demo", dst)


"""
双边滤波器顾名思义比高斯滤波多了一个高斯方差 σ － d \sigma－dσ－d，
它是基于空间分布的高斯滤波函数，所以在边缘附近，离的较远的像素不会太多影响到边缘上的像素值，
这样就保证了边缘附近像素值的保存。但是由于保存了过多的高频信息，对于彩色图像里的高频噪声，
双边滤波器不能够干净的滤掉，只能够对于低频信息进行较好的滤波
均值迁移滤波器，主要的效果主要是使得图片具有油画效果，也就是图片中的边缘得以保留，但是差异一定范围内的像素点将展现区域内的平均值
bilateralFilter(src, d, sigmaColor, sigmaSpace[, dst[, borderType]]) -> dst
  - src: 输入图像。
  - d:   在过滤期间使用的每个像素邻域的直径。如果输入d非0，则sigmaSpace由d计算得出，如果sigmaColor没输入，则sigmaColor由sigmaSpace计算得出。
  - sigmaColor: 色彩空间的标准方差，一般尽可能大。
                较大的参数值意味着像素邻域内较远的颜色会混合在一起，
                从而产生更大面积的半相等颜色。
  - sigmaSpace: 坐标空间的标准方差(像素单位)，一般尽可能小。
                参数值越大意味着只要它们的颜色足够接近，越远的像素都会相互影响。
                当d > 0时，它指定邻域大小而不考虑sigmaSpace。 
                否则，d与sigmaSpace成正比。

"""
def set_bi(image):      #双边滤波
    dst = cv.bilateralFilter(image, 0, 100, 5)
    cv.imshow("bi_demo", dst)


def set_shift(image):   #均值迁移
    dst = cv.pyrMeanShiftFiltering(image, 10, 50)
    cv.imshow("shift_demo", dst)


"""-------------------锐化------------------------"""
"""
锐化滤波
拉普拉斯算子锐化与sobel算子锐化
"""
def sharp_laplace(image): # 拉普拉斯算子锐化
    # 拉普拉斯算子锐化
    kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]], np.float32)  # 定义拉普拉斯算子
    dst = cv.filter2D(image, -1, kernel=kernel)
    cv.imshow("laplace_sharp", dst)


def sharp_sobel(image): # sobel算子锐化
    # gimg = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    gimg = image
    # 对x方向梯度进行sobel边缘提取
    x = cv.Sobel(gimg, cv.CV_64F, 1, 0)
    # 对y方向梯度进行sobel边缘提取
    y = cv.Sobel(gimg, cv.CV_64F, 0, 1)
    # 对x方向转回uint8
    absX = cv.convertScaleAbs(x)
    # 对y方向转会uint8
    absY = cv.convertScaleAbs(y)
    # x，y方向合成边缘检测结果
    dst1 = cv.addWeighted(absX, 0.5, absY, 0.5, 0)
    res = dst1 + image
    cv.imshow("label", res)


if __name__ == "__main__":
    cv.imshow("src_image", img)
    # set_median_blur(img)
    # sharp_laplace(img)
    sharp_sobel(img)
    cv.waitKey(0)
    cv.destroyAllWindows()
