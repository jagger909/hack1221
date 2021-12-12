import cv2
import numpy as np
import requests
import random

font                   = cv2.FONT_HERSHEY_SIMPLEX
bottomLeftCornerOfText = (100,100)
fontScale              = 3
fontColor              = (0,220,140)
thickness              = 10
lineType               = 2

def tile(img, x, y):
    im1_s = cv2.resize(img, dsize=(0, 0), fx=0.2, fy=0.2)
    t = np.tile(img, (x, y, 1));

    return t

def show_img(class_num):
    x = random.randint(2, 3)
    y = random.randint(4, 7)
    url = f"https://cocodataset.org/images/cocoicons/{class_num}.jpg"
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