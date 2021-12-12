import cv2
import numpy as np
import requests
import random

CLASSES = {'person':1,
'bicycle':2,
'car':3,
'motorbike':4,
'aeroplane':5,
'bus':6,
'train':7,
'truck':8,
'boat':9,
'traffic light':10,
'fire hydrant':11,
'stop sign':13,
'parking meter':14,
'bench':15,
'bird':16,
'cat':17,
'dog':18,
'horse':19,
'sheep':20,
'cow':21,
'elephant':22,
'bear':23,
'zebra':24,
'giraffe':25,
'backpack':27,
'umbrella':28,
'handbag':31,
'tie':32,
'suitcase':33,
'frisbee':34,
'skis':35,
'snowboard':36,
'sports ball':37,
'kite':38,
'baseball bat':39,
'baseball glove':40,
'skateboard':41,
'surfboard':42,
'tennis racket':43,
'bottle':44,
'wine glass':46,
'cup':47,
'fork':48,
'knife':49,
'spoon':50,
'bowl':51,
'banana':52,
'apple':53,
'sandwich':54,
'orange':55,
'broccoli':56,
'carrot':57,
'hot dog':58,
'pizza':59,
'donut':60,
'cake':61,
'chair':62,
'sofa':63,
'pottedplant':64,
'bed':65,
'diningtable':67,
'toilet':70,
'tvmonitor':72,
'laptop':73,
'mouse':74,
'remote':75,
'keyboard':76,
'cell phone':77,
'microwave':78,
'oven':79,
'toaster':80,
'sink':81,
'refrigerator':82,
'book':84,
'clock':85,
'vase':86,
'scissors':87,
'teddy bear':88,
'hair drier':89,
'toothbrush':90}

font                   = cv2.FONT_HERSHEY_SIMPLEX
bottomLeftCornerOfText = (10,130)
fontScale              = 2
fontColor              = (0,0,255)
thickness              = 10
lineType               = 2

def tile(img, x, y):
    im1_s = cv2.resize(img, dsize=(0, 0), fx=0.2, fy=0.2)
    t = np.tile(img, (x, y, 1));

    return t

def show_img(class_num):
    if not class_num in CLASSES:
        return None
    c = CLASSES[class_num]
    x = random.randint(4, 6)
    y = random.randint(3, 6)
    url = f"https://cocodataset.org/images/cocoicons/{c}.jpg"
    resp = requests.get(url, stream=True).raw
    image = np.asarray(bytearray(resp.read()), dtype="uint8")
    im1 = cv2.imdecode(image, cv2.IMREAD_COLOR)
    im = tile(im1, x, y)
    
    cv2.putText(im, f'{x}x{y}={x*y}', 
        bottomLeftCornerOfText, 
        font, 
        fontScale,
        fontColor,
        thickness,
        lineType)

    return im