#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().system('pip install ultralytics')


# In[2]:


import os
import cv2
import numpy as np
from ultralytics import YOLO


# In[7]:


model = YOLO('yolov8n-cls.pt')
result = model.train(data=r'C:\Users\user\Documents\app\data', epochs=40, imgsz = 255)


# In[9]:


import os
import cv2
import numpy as np
from ultralytics import YOLO
# model = YOLO('yolov8n-cls.pt')
# result = model.train(data='D:\data', epochs=3, imgsz = 255)

model = YOLO(r'C:\Users\user\runs\classify\train7\weights\best.pt')

img = cv2.imread(r'C:\Users\user\Desktop\images\normal skin.jpg')
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


# In[ ]:





# In[ ]:




