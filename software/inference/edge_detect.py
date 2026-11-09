#coding=UTF-8
from pycoral.adapters import common
from pycoral.adapters import detect
from pycoral.utils.dataset import read_label_file
from pycoral.utils.edgetpu import make_interpreter


from PIL import Image
from PIL import ImageDraw
import cv2
import numpy
import requests

import time 

model = 'efficientdet_edgetpu.tflite' #輸入tflite檔
label = 'label.txt'             #輸入標籤檔

cap = cv2.VideoCapture(0)
print('ok')


def draw_objects(draw, objs, labels):
  """Draws the bounding box and label for each object."""
  for obj in objs:
    bbox = obj.bbox
    draw.rectangle([(bbox.xmin, bbox.ymin), (bbox.xmax, bbox.ymax)],
                   outline='red')
    draw.text((bbox.xmin + 10, bbox.ymin + 10),
              '%s\n%.2f' % (labels.get(obj.id, obj.id), obj.score),
              fill='red')


def main():

  labels = read_label_file(label)
  engine = make_interpreter(model)
  engine.allocate_tensors()
  

  while(cap.isOpened()):
    ret, frame = cap.read()  #讀Camera
    start_time = time.time() #開始計時
    image = Image.fromarray(cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)) #Opencv轉PIL
    
    draw = ImageDraw.Draw(image)  #畫框框
    
    
    _, scale = common.set_resized_input(engine, image.size, lambda size: image.resize(size, Image.ANTIALIAS))
    engine.invoke()
    objs = detect.get_objects(engine, 0.4, scale)
    
    for obj in objs:
        print(labels.get(obj.id, obj.id))
        print('  id:    ', obj.id)
        print('  score: ', obj.score)
        print('  bbox:  ', obj.bbox)
    draw_objects(ImageDraw.Draw(image), objs, labels)

    if not objs:
      print('No objects detected.')
    image = cv2.cvtColor(numpy.asarray(image),cv2.COLOR_RGB2BGR)    #PIL轉Opencv
    cv2.imshow('img',image)
    end_time = time.time()  #結束計時
    print('FPS=',1/(end_time-start_time))   #FPS算法
    
    if cv2.waitKey(1) & 0xFF == ord('q'):   #案q離開
      break
  cap.release()
  cv2.destroyAllWindows()


if __name__ == '__main__':
  main()


