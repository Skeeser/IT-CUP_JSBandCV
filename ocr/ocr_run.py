import cv2
import mediapipe as mp
from PIL import Image, ImageDraw, ImageFont
import time
import numpy as np
import os
import math
# import pytesseract
import pyttsx3

"""图像预处理部分"""
class Process:
    def __init__(self):
        pass


"""OCR部分"""
class OcrClass:
    def __init__(self):
        pass


"""语音部分"""
class VoiceOfOcr:
    def __init__(self):
        self.pp = pyttsx3.init()

        # todo 可以设置音量 语调 速度那些
        # 设置音量
        self.vol = self.pp.getProperty('volume')
        # self.pp.setProperty('vol', self.vol + 0.5)

    def sayaddtext(self, text):
        self.pp.say(text)

    def saystart(self):
        self.pp.runAndWait()


"""绘图部分"""
class DrawInFrame:
    def __init__(self):
        # 识别模式： 无， 双手（框范围）， 单手（识别手势数字）
        self.hand_mode = 'None'
        self.hand_num = 0
        # 坐标
        self.last_finger_x = {'Left': 0, 'Right': 0}
        self.last_finger_y = {'Left': 0, 'Right': 0}
        self.last_finger_arc_degree = {'Left': 0, 'Right': 0}
        self.handedness_color = {'Left': (55, 200, 100), 'Right': (255, 10, 155)}
        # 时间
        now = time.time()
        self.stop_time = {'Left': now, 'Right': now}
        self.activate_duration = 0.3
        # 单手
        # 单手触发识别时间
        self.right_hand_circle_list = []
        self.single_hand_duration = 1
        self.single_hand_last_time = None
        self.last_thumb_img = None
        # 手指浮动允许范围
        self.float_distance = 10
        # 导入相关功能
        self.pp_ocr = OcrClass()
        self.pp_voice = VoiceOfOcr()
        # 上次检测结果
        # finger_num 2: OCR
        self.last_detect_res = {'detection': None, 'ocr': '无'}

    def frameaddtext(self, img, text, position, textColor=(0, 255, 0), textSize=30):
        if (isinstance(img, np.ndarray)):  # 判断是否OpenCV图片类型
            img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(img)
        fontStyle = ImageFont.truetype(
            "./fonts/simsun.ttc", textSize, encoding="utf-8")
        draw.text(position, text, textColor, font=fontStyle)
        return cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)

    def generateocrarea(self, ocr_text, line_text_num, line_num, x, y, w, h, frame):
        sub_img = frame[y:y + h, x:x + w]
        green_rect = np.ones(sub_img.shape, dtype=np.uint8) * 0
        res = cv2.addWeighted(sub_img, 0.5, green_rect, 0.5, 1.0)
        for i in range(line_num):
            text = ocr_text[(i * line_text_num):(i + 1) * line_text_num]
            res = self.frameaddtext(res, text, (10, 30 * i + 10), textColor=(255, 255, 255), textSize=18)
        return res

    def generatelabelarea(self, text, x, y, w, h, frame):
        sub_img = frame[y:y + h, x:x + w]
        green_rect = np.ones(sub_img.shape, dtype=np.uint8) * 0
        res = cv2.addWeighted(sub_img, 0.5, green_rect, 0.5, 1.0)
        res = self.frameaddtext(res, text, (10, 10), textColor=(255, 255, 255), textSize=30)
        return res

    # 生成缩略图
    def generateThumb(self, raw_img, frame):
        if self.last_detect_res['detection'] == None:
            self.last_detect_res['detection'] = ['无', 'None']

        frame_height, frame_width, _ = frame.shape
        raw_img_h, raw_img_w, _ = raw_img.shape

        thumb_img_w = 300
        thumb_img_h = math.ceil(raw_img_h * thumb_img_w / raw_img_w)
        thumb_img = cv2.resize(raw_img, (thumb_img_w, thumb_img_h))

        rect_weight = 4
        # 在缩略图上画框
        thumb_img = cv2.rectangle(thumb_img, (0, 0), (thumb_img_w, thumb_img_h), (0, 139, 247), rect_weight)

        # 生成label
        x, y, w, h = (frame_width - thumb_img_w), thumb_img_h, thumb_img_w, 50

        frame = frame.copy()
        frame[y:y + h, x:x + w] = self.generatelabelarea(
            '{label_zh} {label_en}'.format(label_zh=self.last_detect_res['detection'][0],
                                           label_en=self.last_detect_res['detection'][1]), x, y, w, h, frame)
        # OCR todo 根据手指数量判断
        ocr_text = ''
        if self.last_detect_res['detection'] == 2:
            # todo 实现画图ocr的相关函数
            src_im, text_list = self.pp_ocr.ocr_image(raw_img)
            thumb_img = cv2.resize(src_im, (thumb_img_w, thumb_img_h))

            if len(text_list) > 0:
                ocr_text = ''.join(text_list)
                self.last_detect_res['ocr'] = ocr_text
            else:
                self.last_detect_res['ocr'] = '无'
        else:
            # 连着上次检测结果
            ocr_text = self.last_detect_res['ocr']

        frame[0:thumb_img_h, (frame_width - thumb_img_w):frame_width, :] = thumb_img

        # 是否需要显示
        if ocr_text != '' and ocr_text != '无':
            line_text_num = 15
            line_num = math.ceil(len(ocr_text) / line_text_num)
            y, h = (y + h + 20), (32 * line_num)
            frame[y:y + h, x:x + w] = self.generateocrarea(ocr_text, line_text_num, line_num, x, y, w, h, frame)
        self.last_thumb_img = thumb_img
        return frame

    def drawArc(self, frame, point_x, point_y, arc_radius=150, end=360, color=(155, 20, 255), width=20):
        img = Image.fromarray(frame)
        shape = [(point_x - arc_radius, point_y - arc_radius),
                 (point_x + arc_radius, point_y + arc_radius)]
        img1 = ImageDraw.Draw(img)
        img1.arc(shape, start=0, end=end, fill=color, width=width)
        frame = np.asarray(img)
        return frame

    def clearSingleMode(self):
        self.hand_mode = 'None'
        self.last_finger_arc_degree = {'Left': 0, 'Right': 0}
        self.single_hand_last_time = None

    # todo 单手手指数识别函数
    def singleMode(self, x_distance, y_distance, handedness, finger_cord, frame, frame_copy):
        pass

    def checkIndexFingerMove(self, handedness, finger_cord, frame, frame_copy):
        x_distance = abs(finger_cord[0] - self.last_finger_x[handedness])
        y_distance = abs(finger_cord[1] - self.last_finger_y[handedness])

        if self.hand_mode == 'single':
            # 单手模式下遇到双手，释放
            if self.hand_num == 2:
                self.clearSingleMode()
            elif handedness == 'Right':
                frame = self.singleMode(x_distance, y_distance, handedness, finger_cord, frame, frame_copy)

        else:
            # 未移动
            if (x_distance <= self.float_distance) and (y_distance <= self.float_distance):
                # 时间大于触发时间
                if (time.time() - self.stop_time[handedness]) > self.activate_duration:

                    # 画环形图，每隔0.01秒增大5度
                    arc_degree = 5 * ((time.time() - self.stop_time[handedness] - self.activate_duration) // 0.01)
                    if arc_degree <= 360:
                        frame = self.drawArc(
                            frame, finger_cord[0], finger_cord[1], arc_radius=50, end=arc_degree,
                            color=self.handedness_color[handedness], width=15)
                    else:
                        frame = self.drawArc(
                            frame, finger_cord[0], finger_cord[1], arc_radius=50, end=360,
                            color=self.handedness_color[handedness], width=15)
                        self.last_finger_arc_degree[handedness] = 360

                        # 两个手指圆环都满了，直接触发识别
                        if (self.last_finger_arc_degree['Left'] >= 360) and (
                                self.last_finger_arc_degree['Right'] >= 360):
                            rect_l = (self.last_finger_x['Left'], self.last_finger_y['Left'])
                            rect_r = (self.last_finger_x['Right'], self.last_finger_y['Right'])
                            # 外框框
                            frame = cv2.rectangle(frame, rect_l, rect_r, (0, 255, 0), 2)
                            # 框框label
                            if self.last_detect_res['detection']:
                                # 生成label
                                x, y, w, h = self.last_finger_x['Left'], (
                                        self.last_finger_y['Left'] - 50), 120, 50
                                frame = frame.copy()
                                frame[y:y + h, x:x + w] = self.generatelabelarea(
                                    '{label_zh}'.format(label_zh=self.last_detect_res['detection'][0]), x, y, w, h,
                                    frame)

                            # 是否需要重新识别
                            if self.hand_mode != 'double':
                                # 初始化识别结果
                                self.last_detect_res = {'detection': None, 'ocr': '无'}
                                ymin = min(self.last_finger_y['Left'], self.last_finger_y['Right'])
                                ymax = max(self.last_finger_y['Left'], self.last_finger_y['Right'])
                                xmin = min(self.last_finger_x['Left'], self.last_finger_x['Right'])
                                xmax = max(self.last_finger_x['Left'], self.last_finger_x['Right'])
                                # 传给缩略图
                                raw_img = frame_copy[ymin: ymax, xmin: xmax, ]
                                frame = self.generateThumb(raw_img, frame)

                            self.hand_mode = 'double'

                            # 只有右手圆环满，触发描线功能
                        if (self.hand_num == 1) and (self.last_finger_arc_degree['Right'] == 360):
                            self.hand_mode = 'single'
                            self.single_hand_last_time = time.time()  # 记录一下时间
                            self.right_hand_circle_list.append((finger_cord[0], finger_cord[1]))

            else:
                # 移动位置，重置时间
                self.stop_time[handedness] = time.time()
                self.last_finger_arc_degree[handedness] = 0

        self.last_finger_x[handedness] = finger_cord[0]
        self.last_finger_y[handedness] = finger_cord[1]
        return frame


class FingerOcr2Voice:
    def __init__(self):
        # 在不同设备中修改这个设备号， 以及分辨率
        # 摄像头设备号
        self.camera_num = 0
        self.resize_w = 960
        self.resize_h = 720
        # 是否镜像
        self.if_filp = False

        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        self.mp_hands = mp.solutions.hands
        self.image = None

    def checkHandsIndex(self, handedness):
        if len(handedness) == 1:
            handedness_list = ['Left' if handedness[0].classification[0].label == 'Right' else 'Right']
        else:
            handedness_list = [handedness[1].classification[0].label, handedness[0].classification[0].label]
        return handedness_list

    def recognize(self):
        drawInfo = DrawInFrame()
        fpsTime = time.time()

        cap = cv2.VideoCapture(self.camera_num)


        fps = cap.get(cv2.CAP_PROP_FPS)
        fps = 18

        with self.mp_hands.Hands(min_detection_confidence=0.7,
                                 min_tracking_confidence=0.5,
                                 max_num_hands=2) as hands:
            while cap.isOpened():
                success, self.image = cap.read()
                self.image = cv2.resize(self.image, (self.resize_w, self.resize_h))

                # todo 需要根据镜头位置来调整,动态调整文本角度， 注意要隔一段时间调用
                # self.image = cv2.rotate( self.image, cv2.ROTATE_180)

                if not success:
                    print("空帧.")
                    continue

                self.image.flags.writeable = False
                self.image = cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB)

                if self.if_filp:
                 self.image = cv2.flip(self.image, 1)

                # mediapipe模型处理
                results = hands.process(self.image)

                self.image.flags.writeable = True
                self.image = cv2.cvtColor(self.image, cv2.COLOR_RGB2BGR)

                # 保存缩略图
                if isinstance(drawInfo.last_thumb_img, np.ndarray):
                    self.image = drawInfo.generateThumb(drawInfo.last_thumb_img, self.image)

                hand_num = 0
                # 判断是否有手掌
                if results.multi_hand_landmarks:

                    # 记录左右手index
                    handedness_list = self.checkHandsIndex(results.multi_handedness)
                    hand_num = len(handedness_list)

                    drawInfo.hand_num = hand_num

                    # 复制一份干净的原始frame
                    frame_copy = self.image.copy()
                    # 遍历每个手掌
                    for hand_index, hand_landmarks in enumerate(results.multi_hand_landmarks):
                        # 容错
                        if hand_index > 1:
                            hand_index = 1

                        # 在画面标注手指
                        self.mp_drawing.draw_landmarks(
                            self.image,
                            hand_landmarks,
                            self.mp_hands.HAND_CONNECTIONS,
                            self.mp_drawing_styles.get_default_hand_landmarks_style(),
                            self.mp_drawing_styles.get_default_hand_connections_style())

                        # 解析手指，存入各个手指坐标
                        landmark_list = []

                        # 用来存储手掌范围的矩形坐标
                        paw_x_list = []
                        paw_y_list = []
                        for landmark_id, finger_axis in enumerate(
                                hand_landmarks.landmark):
                            landmark_list.append([
                                landmark_id, finger_axis.x, finger_axis.y,
                                finger_axis.z
                            ])
                            paw_x_list.append(finger_axis.x)
                            paw_y_list.append(finger_axis.y)
                        if landmark_list:
                            # 比例缩放到像素
                            ratio_x_to_pixel = lambda x: math.ceil(x * self.resize_w)
                            ratio_y_to_pixel = lambda y: math.ceil(y * self.resize_h)

                            # 设计手掌左上角、右下角坐标
                            paw_left_top_x, paw_right_bottom_x = map(ratio_x_to_pixel,
                                                                     [min(paw_x_list), max(paw_x_list)])
                            paw_left_top_y, paw_right_bottom_y = map(ratio_y_to_pixel,
                                                                     [min(paw_y_list), max(paw_y_list)])

                            # 获取食指指尖坐标
                            index_finger_tip = landmark_list[8]
                            index_finger_tip_x = ratio_x_to_pixel(index_finger_tip[1])
                            index_finger_tip_y = ratio_y_to_pixel(index_finger_tip[2])

                            # 获取中指指尖坐标
                            middle_finger_tip = landmark_list[12]
                            middle_finger_tip_x = ratio_x_to_pixel(middle_finger_tip[1])
                            middle_finger_tip_y = ratio_y_to_pixel(middle_finger_tip[2])

                            # 画x,y,z坐标
                            label_height = 30
                            label_wdith = 130
                            cv2.rectangle(self.image, (paw_left_top_x - 30, paw_left_top_y - label_height - 30),
                                          (paw_left_top_x + label_wdith, paw_left_top_y - 30), (0, 139, 247), -1)

                            l_r_hand_text = handedness_list[hand_index][:1]

                            cv2.putText(self.image,
                                        "{hand} x:{x} y:{y}".format(hand=l_r_hand_text, x=index_finger_tip_x,
                                                                    y=index_finger_tip_y),
                                        (paw_left_top_x - 30 + 10, paw_left_top_y - 40),
                                        cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 255), 2)

                            # 给手掌画框框
                            cv2.rectangle(self.image, (paw_left_top_x - 30, paw_left_top_y - 30),
                                          (paw_right_bottom_x + 30, paw_right_bottom_y + 30), (0, 139, 247), 1)

                            # 释放单手模式
                            line_len = math.hypot((index_finger_tip_x - middle_finger_tip_x),
                                                  (index_finger_tip_y - middle_finger_tip_y))

                            if line_len < 50 and handedness_list[hand_index] == 'Right':
                                drawInfo.clearSingleMode()
                                drawInfo.last_thumb_img = None

                                # 传给画图类，如果食指指尖停留超过指定时间（如0.3秒），则启动画图，左右手单独画
                            self.image = drawInfo.checkIndexFingerMove(handedness_list[hand_index],
                                                                       [index_finger_tip_x, index_finger_tip_y],
                                                                       self.image, frame_copy)

                # 显示刷新率FPS
                cTime = time.time()
                fps_text = 1 / (cTime - fpsTime)
                fpsTime = cTime
                self.image = drawInfo.frameaddtext(self.image, "帧率: " + str(int(fps_text)), (10, 30),
                                                        textColor=(0, 155, 20), textSize=25)
                self.image = drawInfo.frameaddtext(self.image, "手掌: " + str(hand_num), (10, 60),
                                                        textColor=(0, 55, 250), textSize=25)
                self.image = drawInfo.frameaddtext(self.image, "模式: " + str(drawInfo.hand_mode), (10, 90),
                                                        textColor=(156, 155, 50), textSize=25)

                # 显示画面
                # self.image = cv2.resize(self.image, (resize_w//2, resize_h//2))
                cv2.imshow('virtual reader', self.image)
                if cv2.waitKey(5) & 0xFF == 27:
                    break
            cap.release()


if __name__ == "__main__":
    fo2v = FingerOcr2Voice()
    fo2v.recognize()
