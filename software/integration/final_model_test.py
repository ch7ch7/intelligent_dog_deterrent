import cv2
import numpy as np
import time
from pycoral.adapters import common
from pycoral.adapters import detect
from pycoral.utils.dataset import read_label_file
from pycoral.utils.edgetpu import make_interpreter
from PIL import Image
from PIL import ImageDraw
import threading

model = 'efficientdet.tflite' #輸入tflite檔
label = 'labels.txt'             #輸入標籤檔


def draw_objects(draww, objs, labels):
  """Draws the bounding box and label for each object."""
  for obj in objs:
    bbox = obj.bbox
    draww.rectangle([(bbox.xmin, bbox.ymin), (bbox.xmax, bbox.ymax)],
                   outline='red')
    draww.text((bbox.xmin + 10, bbox.ymin + 10),
              '%s\n%.2f' % (labels.get(obj.id, obj.id), obj.score),
              fill='red')



        
def get_iou(ground_truth, pred):
    ground_truth = np.array(ground_truth, dtype=np.float32)
    pred = np.array(pred, dtype=np.float32)
    # coordinates of the area of intersection.
    ix1 = np.maximum(ground_truth[0], pred[0])
    iy1 = np.maximum(ground_truth[1], pred[1])
    ix2 = np.minimum(ground_truth[2], pred[2])
    iy2 = np.minimum(ground_truth[3], pred[3])
     
    # Intersection height and width.
    i_height = np.maximum(iy2 - iy1 + 1, np.array(0.))
    i_width = np.maximum(ix2 - ix1 + 1, np.array(0.))
     
    area_of_intersection = i_height * i_width
     
    # Ground Truth dimensions.
    gt_height = ground_truth[3] - ground_truth[1] + 1
    gt_width = ground_truth[2] - ground_truth[0] + 1
     
    # Prediction dimensions.
    pd_height = pred[3] - pred[1] + 1
    pd_width = pred[2] - pred[0] + 1
     
    area_of_union = gt_height * gt_width + pd_height * pd_width - area_of_intersection
    pr = pd_height*pd_width
     
    # iou = area_of_intersection / area_of_union
    iou = area_of_intersection / pr
    return iou

def show_xy(event,x,y,flags,userdata):
    global draw,flag_mouse_lbutton_down, point_x, point_y, temp_image, box, frame
    #print(event,x,y,flags)

    if draw:
        print("drawing")
        if event == cv2.EVENT_LBUTTONDOWN:
            # print("EVENT_LBUTTONDOWN", event, x, y, flags, param)
            flag_mouse_lbutton_down = True
            point_x = x
            point_y = y

        if event == cv2.EVENT_LBUTTONUP:
            # print("EVENT_LBUTTONUP", event, x, y, flags, param)
            flag_mouse_lbutton_down = False
            cv2.rectangle(frame, (point_x, point_y), (x, y), (0, 0, 255), 2)
            temp_image = frame.copy()   #畫多重方框
            #print(f'point_x:{point_x}  point_y:{point_y}   x:{x}  y:{y}')
            box.append([point_x, point_y, x, y])
        if event == cv2.EVENT_MOUSEMOVE and flags == cv2.EVENT_FLAG_LBUTTON:
            frame = temp_image.copy()
            cv2.rectangle(frame, (point_x, point_y), (x, y), (0, 0, 255), 2)
    
    if len(box)==1:
        draw = False
    if not draw:
        if(x >= 0 and x <= frame.shape[0]//5):
            if(y >= 0 and y <= frame.shape[0]//5):
                if(flags == 1):
                    print("drow")
                    draw = True
                    time.sleep(0.5)
                    
def job():
    global count_num, dog_gogo
    time_start = time.time()
    while True:
        print("counting!!!!!!!!!!!!!!!!!!!!!!!")
        if (time_start-time.time())>1:
            dog_gogo = True
        if not count_num:
            thread_start = False
            return
        time.sleep(0.05)
        print("test threading function")
    thread_start = False
    dog_gogo = True
    return

def shocking_dog():
    global t_shock_start,dog_gogo
    for i in range(3):
        print("shocking!!!!!!!!!!!!!!!!!!!!!")
        button = "on"
        time.sleep(0.5)
    t_shock_start = False
    dog_gogo = False
            
            
t_count = threading.Thread(target = job)
t_count.setDaemon(True)
t_shock = threading.Thread(target = shocking_dog)
t_shock.setDaemon(True)

flag_mouse_lbutton_down = False
t_shock_start = False
thread_start = False
dog_gogo = False
count_num = False
ff = True
draw = False
point_x = -1
point_y = -1
box = []

cap = cv2.VideoCapture(0)
ret,frame = cap.read()


cv2.namedWindow("image")


while True:   # screen showing and choose image, then stop loop
    ret,frame = cap.read()
    output = cv2.rectangle(frame, (0,0), (frame.shape[0]//5,frame.shape[0]//5), (0,255,0), 5)
    cv2.putText(output, "drow", (frame.shape[0]//40,frame.shape[0]//20), cv2.FONT_HERSHEY_TRIPLEX, 0.5, (0, 0, 255), 1, 4)
    cv2.imshow('image',output)
    cv2.setMouseCallback("image", show_xy)
    time.sleep(0.02)
    if draw:
        ret,frame = cap.read()
        temp_image = frame.copy()
        break
        #time.sleep(2)
    if cv2.waitKey(10) == ord("q"):
        break
    
while True:   # screen drawing rectangle loop
    output = cv2.rectangle(frame, (0,0), (frame.shape[0]//5,frame.shape[0]//5), (0,255,0), 5)
    cv2.putText(frame, "drow", (frame.shape[0]//40,frame.shape[0]//20), cv2.FONT_HERSHEY_TRIPLEX, 0.5, (0, 0, 255), 1, 4)
    cv2.imshow('image',frame)
    cv2.setMouseCallback("image", show_xy)
    time.sleep(0.02)
    if len(box)==1 and ff:
        ff = False
        #print(get_iou(box[0], box[1]))
        break
    if cv2.waitKey(10) == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
print(f'point_x:{box[0][0]}  point_y:{box[0][1]}   x:{box[0][2]}  y:{box[0][3]}')

# load model then start detect
print("\n\nLoading Model, please wait 1 second !!\n")

labels = read_label_file(label)  #read label
engine = make_interpreter(model) #read model
engine.allocate_tensors()



cap = cv2.VideoCapture(0)
ret,frame = cap.read()
print("start detect")

while(cap.isOpened()):   # Model detect loop
    ret, frame = cap.read()  #讀Camera
    start_time = time.time() #開始計時
    image = Image.fromarray(cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)) #Opencv轉PIL
    
    draw = ImageDraw.Draw(image)  #畫框框
    
    _, scale = common.set_resized_input(engine, image.size, lambda size: image.resize(size, Image.ANTIALIAS))
    engine.invoke()
    objs = detect.get_objects(engine, 0.4, scale)
    
    for obj in objs:
        x = [obj.bbox.xmin, obj.bbox.ymin, obj.bbox.xmax, obj.bbox.ymax]
        print(labels.get(obj.id, obj.id))
        print('  id:    ', obj.id)
        print('  score: ', obj.score)
        print('  bbox:  ', obj.bbox)
        print(f'IOU: {get_iou(box[0], x)}')
        if obj.id==3:
            if get_iou(box[0], x)<0.5:
                continue
            if not thread_start:
                t_count.start()
                t_count_start = True
            count_num = True
            print("people detect")
    draw_objects(ImageDraw.Draw(image), objs, labels)

    if not objs:
        print('No objects detected.')
    image = cv2.cvtColor(np.asarray(image),cv2.COLOR_RGB2BGR)    #PIL轉Opencv
    cv2.rectangle(image, (box[0][0], box[0][1]), (box[0][2], box[0][2]), (0, 255, 0), 2)
    cv2.imshow('img',image)
    if dog_gogo:
        if not t_shock_start:
            t_shock.start()
            t_shock_start = True
    end_time = time.time()  #結束計時
    print('FPS=',1/(end_time-start_time))   #FPS算法

    if cv2.waitKey(1) & 0xFF == ord('q'):   #案q離開
        break
cap.release()
cv2.destroyAllWindows()

