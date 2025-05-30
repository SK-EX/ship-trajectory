import cv2 as cv

video = cv.VideoCapture('test.mp4')

#ret 获取第一针图像 ret 为boolean值，表示是否获取到，frame表示帧图像
ret,frame = video.read()
cv.namedWindow("Demo", cv.WINDOW_AUTOSIZE)
# 可以在图片上选择roi区域(roi 感兴趣区域)
x, y, w, h = cv.selectROI("Demo", frame, True, False)
track_window = (x, y, w, h)

# 获取ROI直方图
roi = frame[y:y+h, x:x+w]
hsv_roi = cv.cvtColor(roi, cv.COLOR_BGR2HSV)

# inRange函数设置亮度阈值
# 去除低亮度的像素点的影响
# 将低于和高于阈值的值设为0
mask = cv.inRange(hsv_roi, (26, 43, 46), (34, 255, 255))

# 计算直方图,参数为 图片(可多)，通道数，蒙板区域，直方图长度，范围
roi_hist = cv.calcHist([hsv_roi],[0],mask,[180],[0,180])

# 归一化，像素值区间[0,255]
cv.normalize(roi_hist,roi_hist,0,255,cv.NORM_MINMAX)

# 设置迭代的终止标准，最多十次迭代
term_crit = ( cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 10, 1 )

while True:
    ret, frame = video.read()
    if ret is False:
        break
    hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV)

    # 直方图反向投影
    dst = cv.calcBackProject([hsv],[0],roi_hist,[0,180],1)

    # 均值迁移，搜索更新roi区域
    ret, track_window = cv.meanShift(dst, track_window, term_crit)

    # 绘制窗口
    x,y,w,h = track_window
    cv.rectangle(frame, (x,y), (x+w,y+h), 255,2)
    cv.imshow('Demo',frame)
    k = cv.waitKey(60) & 0xff
    if k == 27:
        break
    else:
        cv.imwrite(chr(k)+".jpg",frame)
cv.destroyAllWindows()
video.release()