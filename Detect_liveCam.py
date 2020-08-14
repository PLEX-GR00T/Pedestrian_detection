from __future__ import print_function
from imutils.object_detection import non_max_suppression
from imutils.video import VideoStream
from imutils import paths
import argparse
import numpy as np
import imutils
import time
import cv2

# ap = argparse.ArgumentParser()
# ap.add_argument("-o", "--output", type=str,
#     help="Path to optional output video file")
# args = vars(ap.parse_args())

hog = cv2.HOGDescriptor()
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

print("[INFO] Starting Video stream...")
vs = cv2.VideoCapture(0)
time.sleep(2.0)

if vs.isOpened():
    ret,frame = vs.read()
else:
    ret = False
writer = None
W = None
H = None

while ret:
    
    #frame = vs.read() 
    ret,frame = vs.read()
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    rgb = imutils.resize(frame,width=750)
    orig = frame.copy()
    
    if W is None or H is None:
        (H,W) = frame.shape[:2]

    # if args["output"] is not None and Writer is None:
    #     fourcc = cv2.VideoWriter_fourcc(*"MJPG")
    #     writer + cv2,VideoWriter(args["output"], fourcc, 30
    #         (W,H), True)
    
    (rects, weights) = hog.detectMultiScale(frame, winStride=(4,4),
        padding=(8,8), scale= 1.05)

    for(x, y, w, h) in rects:
        cv2.rectangle(orig, (x,y), (x+w, y+h), (0,0,255), 2)

    rects = np.array([[x, y, x+w, y+h] for (x, y, w, h) in rects])
    pick = non_max_suppression(rects, probs= None, overlapThresh=0.65)

    for (xA,yA,xB,yB) in pick:
        cv2.rectangle(rgb, (xA,yA), (xB, yB), (0,255,0), 2)

    cv2.imshow("Before NMS", orig)
    cv2.imshow("After NMS", rgb)
    if cv2.waitKey(1) & 0xFF == ord('q'): 
        break

cv2.destroyAllWindows()
vs.stop()



                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                    