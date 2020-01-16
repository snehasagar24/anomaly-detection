# -*- coding: utf-8 -*-
"""
Created on Fri Jan 10 11:27:01 2020

@author: sneha.sagar
"""

# -*- coding: utf-8 -*-
"""
Created on Tue Dec  3 17:05:05 2019

@author: rahul.tripathi
"""

import numpy as np
import cv2

# define the lower and upper boundaries of the colors in the HSV color space
lower = {'red':(170, 100, 0),
     #'green':(35,21,62),
     'blue':(110,50,50),
     #'yellow':(23,59,119)
     
     }

upper = {'red':(180,255,255), 
     #'green':(55,255,255),
     'blue':(130,255,255),
     #'yellow':(43,255,255)
     }

# define standard colors for circle around the object
colors = {'red':(0,0,255),
      'green':(0,255,0),
      'blue':(255,0,0),
      'yellow':(0,255,217)
      }

font = cv2.FONT_HERSHEY_SIMPLEX

cv2.namedWindow('Anamoly Detection', cv2.WINDOW_NORMAL)
while True:
    cam = cv2.VideoCapture(1)
    #fourcc = cv2.VideoWriter_fourcc(*'MJPG')
    #out = cv2.VideoWriter('output.avi',fourcc,20.0,(640,480),True)
    ret, frame = cam.read()
    
    
    
    blurred = cv2.GaussianBlur(frame,(11,11),0)
    hsv = cv2.cvtColor(blurred,cv2.COLOR_BGR2HSV)
    
    for key, value in upper.items():
        kernel = np.ones((9,9),np.uint8)
        mask = cv2.inRange(hsv,lower[key],upper[key])
        mask = cv2.morphologyEx(mask,cv2.MORPH_OPEN,kernel)
        mask = cv2.morphologyEx(mask,cv2.MORPH_CLOSE,kernel)
    
        #Calculate percentage of pixel colors
        output = cv2.countNonZero(mask)
        res = np.divide(float(output),mask.shape[0]*int(mask.shape[1] / 128))
        percent_colors = np.multiply((res),400) / 10000
        percent=(np.round(percent_colors*100,2))
        if (percent <30.00):
            msg="Anomaly Detected"
        else:
            msg= " "
        cnts = cv2.findContours(mask.copy(),cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)[-2]
        center = None
    
        if len(cnts) > 0:
            c = max(cnts, key=cv2.contourArea)
            ((x,y), radius) = cv2.minEnclosingCircle(c)
            M = cv2.moments(c)
            center = (int(M["m10"] / M["m00"]),int(M["m01"] / M["m00"]))
            if radius > 0.1:
                cv2.circle(frame,(int(x),int(y)),int(radius),colors[key],2)
                cv2.putText(frame,
                            msg ,
                            (int(x-radius),int(y-radius)),
                            font,
                            0.6,
                           (0, 0, 0 ),
                            2)
    cv2.imshow("Anamoly Detection",frame)
    if(cv2.waitKey(1) & 0xFF == ord('q')):
        break
cam.release()
cv2.destroyAllWindows()
