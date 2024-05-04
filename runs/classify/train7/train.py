import os
import cv2
import numpy as np
from ultralytics import YOLO
import probs
# model = YOLO('yolov8n-cls.pt')
# result = model.train(data='D:\data', epochs=3, imgsz = 255)

model = YOLO(r'C:\Users\MASTERS\Desktop\Final_Project\runs\classify\train\weights\best.pt')

img = cv2.imread('istockphoto-1398811731-612x612.jpg')
imS = cv2.resize(img,(255,255))
results = model(imS, show = True)
names_dict = results[0].names
probs = results[0].probs.data.tolist()
print(names_dict)
print(probs)
if max(probs) < 0.99:
    prediction = None
    print("You don't have any disease")
else:
    prediction = names_dict[probs.index(max(probs))]
    print(f'We regret to inform you that you have {prediction} disease ')



cv2.waitKey()
cv2.destroyAllWindows

