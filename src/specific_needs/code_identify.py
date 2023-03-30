import cv2 as cv
import numpy as np
import time
from pyzbar.pyzbar import decode

img = cv.imread("C:\\mydoc\\ElectricDesign_project\\nobodyfly\\IT-CUP_JSBandCV\\test_pic\\42.jpg", cv.IMREAD_COLOR)

cap = cv.VideoCapture(0)
cap.set(3, 640)
cap.set(4, 480)

while True:
    success, img = cap.read()
    for barcode in decode(img):
        codeData = barcode.data.decode('utf-8')
        print(time.strftime("%H:%M:%S-") + codeData)
        pts = np.array([barcode.polygon], np.int32)
        pts = pts.reshape((-1, 1, 2))
        cv.polylines(img, [pts], True, (255, 0, 255), 5)
        pts2 = barcode.rect
        cv.putText(img, time.strftime("%H:%M:%S-") + codeData, (pts2[0], pts2[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.9,
                    (255, 0, 255), 2)

    cv.imshow('image', img)
    cv.waitKey(1)

