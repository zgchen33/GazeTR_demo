import os
from tqdm import tqdm
import numpy as np
import cv2
import csv

folder_path = os.getcwd()
point = dict()

gaze_x = []
gaze_y = []
with open("bc.csv", newline='') as csvfile:
    reader = csv.reader(csvfile, delimiter=',', quotechar='"')
    for row in reader:
        gaze_x.append(row[0])
        gaze_y.append(row[1])

length = len(gaze_x)


for file_name in tqdm(os.listdir(folder_path)):

    if file_name.endswith('.npy'):
        point[os.path.splitext(file_name)[0]] = np.load(file_name)

total_frame = 374


video_path = os.path.join(folder_path, "sucai")
video_frame = []
for file_name in os.listdir(video_path):
    if file_name.endswith("jpg"):
        video_frame.append(file_name)
video_frame.sort()
rate = 1.3
cv2.namedWindow('lol', 0)
cv2.resizeWindow('lol', int(1920 / rate), int(1080 / rate))
cnt = 0
for frame_name in video_frame:
    # 检查文件是否为图像文件
    if frame_name.endswith('.jpg'):
        # 读取图像文件
        file_name = os.path.join(video_path, frame_name)
        img = cv2.imread(os.path.join(folder_path, file_name))
        # 播放图像

        gaze_frame = int(cnt * length / 374)
        cv2.circle(img, (int(gaze_x[gaze_frame]), int(gaze_y[gaze_frame])), 30, (200, 200, 0), 5)
        d = 2147483647
        flag = -1
        for key in point:
            if point[key][cnt][0] != -1:
                ds = min(pow(abs(int(point[key][cnt][0]) - int(gaze_x[gaze_frame])),2) +
                         pow(abs(int(point[key][cnt][1]) - int(gaze_y[gaze_frame])),2), d)
                if ds < d:
                    flag = key
                    d = ds
                cv2.circle(img, (int(point[key][cnt][0]), int(point[key][cnt][1])), 30, (0, 0, 255), 5)
        print(flag)
        cv2.circle(img, (int(point[flag][cnt][0]), int(point[flag][cnt][1])), 30, (0, 255,), 5)
        cv2.line(img, (int(point[flag][cnt][0]), int(point[flag][cnt][1])),
                 (int(gaze_x[gaze_frame]), int(gaze_y[gaze_frame])),(200, 200, 0),5)
        cv2.imshow('lol', img)
        cnt = cnt + 1
        # 等待按下任意按键，持续显示图像直到按键被按下
        cv2.waitKey()

# 关闭窗口
cv2.destroyAllWindows()
