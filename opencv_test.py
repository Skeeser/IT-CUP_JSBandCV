import cv2 as cv
import numpy as np

capture = cv.VideoCapture(11)
# capture.set(cv.CAP_PROP_FRAME_WIDTH, 4224)
# capture.set(cv.CAP_PROP_FRAME_HEIGHT, 3136)
capture.set(cv.CAP_PROP_FPS, 30)
# 亮度
# capture.set(cv.CAP_PROP_BRIGHTNESS, 50)
# 对比度
# capture.set(cv.CAP_PROP_CONTRAST, 50)
# 饱和度
# capture.set(cv.CAP_PROP_SATURATION, 50)
# 增益
# capture.set(cv.CAP_PROP_GAIN, 50)

# 适应屏幕大小
def scal_screen(frame, sceen_width, sceen_height):
    height, width = frame.shape[:2]
    # print("height:", height," width:", width)
    if height > width:
        scale = min(1, sceen_height / height)
    else:
        scale = min(1, sceen_width / width)
    frame = cv.resize(frame,None, fx=scale, fy=scale, interpolation=cv.INTER_LINEAR)
    return frame

def video_demo():
    
    while(True):
        ret, frame = capture.read()
        frame = scal_screen(frame, 720, 576)
        cv.imshow("video", frame)
        k = cv.waitKey(1)
        if k == 27:
            break
            
        
video_demo()
cv.waitKey()
cv.destroyAllWindows()