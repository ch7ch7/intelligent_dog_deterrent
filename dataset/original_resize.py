import os
import cv2

img_list = []
print(os.getcwd())
os.mkdir("img")
for i in os.listdir(os.getcwd()):
    if i.endswith('jpg') or i.endswith('png'):
        img = cv2.imread(i)
        img = cv2.resize(img, (640,480))
        cv2.imwrite(os.path.join("img", i), img)