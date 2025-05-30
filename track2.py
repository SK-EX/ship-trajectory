from itertools import count
import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch



#这个是轨迹检测文件
#右键点击运行

# 定义视频文件路径
video_path = "test3.mp4"

# 打开视频文件，然后热加载权重模型，
cap = cv2.VideoCapture(video_path)

# 创建背景减除器（这里使用MOG2算法，可根据需求选择其他算法）
fgbg = cv2.createBackgroundSubtractorKNN()

# 用于存储物体轨迹点的列表
trajectory = []

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # 应用背景减除器获取前景掩码
    fgmask = fgbg.apply(frame)

    # 进行形态学操作以去除噪声
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_OPEN, kernel)

    # 查找轮廓 ， contours返回物体轮廓值
    contours, _ = cv2.findContours(fgmask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for contour in contours:
        if cv2.contourArea(contour) > 3000:  # 根据面积筛选物体，可调整阈值
            # 获取物体的外接矩形框
            x, y, w, h = cv2.boundingRect(contour)
            # 在帧上绘制矩形检测框，这里框的颜色设置为红色，线条粗细为2
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)

            # 计算物体的质心
            M = cv2.moments(contour)
            if M["m00"]!= 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                trajectory.append((cx, cy))
                # 在帧上绘制质心
                cv2.circle(frame, (cx, cy), 5, (0, 255, 0), -1)

    # 绘制轨迹
    for i in range(1, len(trajectory)):
        cv2.line(frame, trajectory[i], trajectory[i - 1], (255, 0, 0), 2)

    cv2.imshow('Frame', frame)
    if cv2.waitKey(30) & 0xFF == ord('q'):
        break

    trajectory_image = np.zeros((frame.shape[0], frame.shape[1], 3))

    for i in range(1, len(trajectory)):
        cv2.line(trajectory_image, trajectory[i], trajectory[i - 1], (255, 0, 0), 2)

cv2.imwrite('result.jpg', trajectory_image)


#轨迹图片保存在result.jpg下
cap.release()
cv2.destroyAllWindows()