import cv2 as cv
import numpy as np

img = cv.imread("C:\\mydoc\\ElectricDesign_project\\nobodyfly\\IT-CUP_JSBandCV\\test_pic\\42.jpg", cv.IMREAD_COLOR)

"""
如何调节参数阈值，看track_bar.py
"""
def patrol_line(image):
    img = cv.GaussianBlur(image, (3, 3), 0)
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    edges = cv.Canny(gray, 50, 150, apertureSize=3)
    lines = cv.HoughLines(edges, 1, np.pi / 180, 110)
    for line in lines:
        rho = line[0][0]
        theta = line[0][1]
        a = np.cos(theta)
        b = np.sin(theta)
        x0 = a * rho
        y0 = b * rho
        x1 = int(x0 + 1000 * (-b))
        y1 = int(y0 + 1000 * (a))
        x2 = int(x0 - 1000 * (-b))
        y2 = int(y0 - 1000 * (a))
        cv.line(img, (x1, y1), (x2, y2), (0, 0, 255), 2)
    cv.imshow("patrol_line", img)


# 推荐用这个，减少计算量，巡线够用
def patrol_line_p(image):
    img = cv.GaussianBlur(image, (3, 3), 0)
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    edges = cv.Canny(gray, 50, 150, apertureSize=3)
    lines = cv.HoughLinesP(edges, 1, np.pi / 180, 30, 300, 5)
    for line in lines:
        x1 = line[0][0]
        y1 = line[0][1]
        x2 = line[0][2]
        y2 = line[0][3]
        cv.line(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
    cv.imshow("patrol_line_p", img)


if __name__ == "__main__":
    cv.imshow("src_image", img)
    # patrol_line(img)
    patrol_line_p(img)
    cv.waitKey(0)
    cv.destroyAllWindows()