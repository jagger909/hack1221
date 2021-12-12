import cv2
import numpy as np
import requests
import random

CLASSES = {1:1,
2:2,
3:3,
4:4,
5:5,
6:6,
7:7,
8:8,
9:9,
10:10,
11:11,
12:13,
13:14,
14:15,
15:16,
16:17,
17:18,
18:19,
19:20,
20:21,
21:22,
22:23,
23:24,
24:25,
25:27,
26:28,
27:31,
28:32,
29:33,
30:34,
31:35,
32:36,
33:37,
34:38,
35:39,
36:40,
37:41,
38:42,
39:43,
40:44,
41:46,
42:47,
43:48,
44:49,
45:50,
46:51,
47:52,
48:53,
49:54,
50:55,
51:56,
52:57,
53:58,
54:59,
55:60,
56:61,
57:62,
58:63,
59:64,
60:65,
61:67,
62:70,
63:72,
64:73,
65:74,
66:75,
67:76,
68:77,
69:78,
70:79,
71:80,
72:81,
73:82,
74:84,
75:85,
76:86,
77:87,
78:88,
79:89,
80:90}

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
    c = CLASSES[int(class_num)]
    x = random.randint(3, 6)
    y = random.randint(3, 6)
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