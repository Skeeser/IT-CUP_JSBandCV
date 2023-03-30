# 1.图像预处理 preprocessing

## （1）灰度化

## 图像灰度化处理一般有以下几种方式：

1. 分量法

将彩色图像中的三分量的亮度作为三个灰度图像的灰度值，可根据应用需要选取一种灰度图像。


　2. 最大值法

将彩色图像中的三分量亮度的最大值作为灰度图的灰度值。


3. 平均值法

将彩色图像中的三分量亮度求平均得到一个灰度值。

4. 加权平均法

根据重要性及其它指标，将三个分量以不同的权值进行加权平均。R、G、B钱面系数即所加权值，可任意改变。



## 	滤波

## 	二值化

## 	边缘检测



# 2.需求实现

#### 1.轮廓检测

```
cv2.findContours(image, mode, method[, contours[, hierarchy[, offset ]]])
```

返回两个值：contours：hierarchy。
**参数**

```
第一个参数是寻找轮廓的图像；

第二个参数表示轮廓的检索模式，有四种（本文介绍的都是新的cv2接口）：
    cv2.RETR_EXTERNAL表示只检测外轮廓
    cv2.RETR_LIST检测的轮廓不建立等级关系
    cv2.RETR_CCOMP建立两个等级的轮廓，上面的一层为外边界，里面的一层为内孔的边界信息。如果内孔内还有一个连通物体，这个物体的边界也在顶层。
    cv2.RETR_TREE建立一个等级树结构的轮廓。

第三个参数method为轮廓的近似办法
    cv2.CHAIN_APPROX_NONE存储所有的轮廓点，相邻的两个点的像素位置差不超过1，即max（abs（x1-x2），abs（y2-y1））==1
    cv2.CHAIN_APPROX_SIMPLE压缩水平方向，垂直方向，对角线方向的元素，只保留该方向的终点坐标，例如一个矩形轮廓只需4个点来保存轮廓信息
    cv2.CHAIN_APPROX_TC89_L1，CV_CHAIN_APPROX_TC89_KCOS使用teh-Chinl chain 近似算法
```

返回值
cv2.findContours()函数返回两个值，一个是轮廓本身，还有一个是每条轮廓对应的属性。
contour返回值
cv2.findContours()函数首先返回一个list，list中每个元素都是图像中的一个轮廓，用numpy中的ndarray表示。这个概念非常重要。在下面drawContours中会看见。通过

```
print (type(contours))
print (type(contours[0]))
print (len(contours))
```

可以验证上述信息。会看到本例中有两条轮廓，一个是五角星的，一个是矩形的。每个轮廓是一个ndarray，每个ndarray是轮廓上的点的集合。
由于我们知道返回的轮廓有两个，因此可通过

```
cv2.drawContours(img,contours,0,(0,0,255),3)
和
cv2.drawContours(img,contours,1,(0,255,0),3)
```

分别绘制两个轮廓，关于该参数可参见下面一节的内容。同时通过

```
print (len(contours[0]))
print (len(contours[1]))
```

输出两个轮廓中存储的点的个数，可以看到，第一个轮廓中只有4个元素，这是因为轮廓中并不是存储轮廓上所有的点，而是只存储可以用直线描述轮廓的点的个数，比如一个“正立”的矩形，只需4个顶点就能描述轮廓了。
hierarchy返回值
此外，该函数还可返回一个可选的hiararchy结果，这是一个ndarray，其中的元素个数和轮廓个数相同，每个轮廓contours[i]对应4个hierarchy元素hierarchy[i][0] ~hierarchy[i][3]，分别表示后一个轮廓、前一个轮廓、父轮廓、内嵌轮廓的索引编号，如果没有对应项，则该值为负数。
通过

```
print (type(hierarchy))
print (hierarchy.ndim)
print (hierarchy[0].ndim)
print (hierarchy.shape)
```

得到

```
3
2
(1, 2, 4)
```

可以看出，hierarchy本身包含两个ndarray，每个ndarray对应一个轮廓，每个轮廓有四个属性。
轮廓的绘制
OpenCV中通过cv2.drawContours在图像上绘制轮廓。 
cv2.drawContours()函数

```
cv2.drawContours(image, contours, contourIdx, color[, thickness[, lineType[, hierarchy[, maxLevel[, offset ]]]]])
```

```
第一个参数是指明在哪幅图像上绘制轮廓；
第二个参数是轮廓本身，在Python中是一个list。
第三个参数指定绘制轮廓list中的哪条轮廓，如果是-1，则绘制其中的所有轮廓。后面的参数很简单。其中thickness表明轮廓线的宽度，如果是-1（cv2.FILLED），则为填充模式。绘制参数将在以后独立详细介绍。
```



#### 2.霍夫线

上面介绍的整个过程在OpenCV中都被封装进了一个函数：cv2.HoughLines()。返回值就是极坐标表示的直线（ρ, θ）。ρ 的单位是像素，θ 的单位是弧度。

```
cv2.HoughLines(image, rho, theta, threshold, lines, sen, stn, min_theta, max_theta)
```

image：输入图像，8-bit灰度图像
rho：生成极坐标时候的像素扫描步长
theta：生成极坐标时候的角度步长
threshold：阈值，只有获得足够交点的极坐标点才被看成是直线
lines：返回值，极坐标表示的直线（ρ, θ）
sen：是否应用多尺度的霍夫变换，如果不是设置0表示经典霍夫变换
stn：是否应用多尺度的霍夫变换，如果不是设置0表示经典霍夫变换
min_theta：表示角度扫描范围最小值
max_theta：表示角度扫描范围最大值
这种方法仅仅是一条直线都需要两个参数，这需要大量的计算。Probabilistic_Hough_Transform 是对霍夫变换的一种优化。它 不会对每一个点都进行计算，而是从一幅图像中随机选取（是不是也可以使用 图像金字塔呢？）一个点集进行计算，对于直线检测来说这已经足够了。但是 使用这种变换我们必须要降低阈值（总的点数都少了，阈值肯定也要小呀！）。

函数如下：

```
cv2.HoughLinesP(image, rho, theta, threshold, lines, minLineLength, maxLineGap)
```

src：输入图像，必须8-bit的灰度图像
rho：生成极坐标时候的像素扫描步长
theta：生成极坐标时候的角度步长
threshold：阈值，只有获得足够交点的极坐标点才被看成是直线
lines：输出的极坐标来表示直线
minLineLength：最小直线长度，比这个短的线都会被忽略。
maxLineGap：最大间隔，如果小于此值，这两条直线 就被看成是一条直线。

