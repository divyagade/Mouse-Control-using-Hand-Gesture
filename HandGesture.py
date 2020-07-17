import cv2
import numpy as np
import math
import wx
from pynput.mouse import Button,Controller
import keyboard

mouse = Controller()
app = wx.App(False)
(sx,sy) = wx.GetDisplaySize()
(camx,camy) = (320,240)
fistflag = 0

cap = cv2.VideoCapture(0)

frame_width = int(cap.get(3))
frame_height = int(cap.get(4))
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('output.avi',fourcc,5,(frame_width,frame_height))

while True:
    ret, img = cap.read()

    cv2.rectangle(img,(300,300),(100,100),(0,255,0),0)
    crop_img = img[100:300,100:300]
    crop_img = cv2.resize(crop_img,(camx,camy))

    grey = cv2.cvtColor(crop_img,cv2.COLOR_BGR2GRAY)

    blur = cv2.GaussianBlur(grey,(15,15),0)

    _,thresh = cv2.threshold(blur,127,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)

    contours, hierarchy = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)

    cnt = max(contours, key = lambda x: cv2.contourArea(x))

    x, y, w, h = cv2.boundingRect(cnt)
    cv2.rectangle(crop_img, (x, y), (x+w, y+h), (0, 0, 255), 0)

    hull = cv2.convexHull(cnt)
    areahull = cv2.contourArea(hull)
    areacnt = cv2.contourArea(cnt)
    arearatio = ((areahull - areacnt) / areacnt) * 100

    drawing = np.zeros(crop_img.shape, np.uint8)
    cv2.drawContours(drawing, [cnt], 0, (0, 255, 0), 0)
    cv2.drawContours(drawing, [hull], 0, (0, 0, 255), 0)

    M = cv2.moments(cnt)
    cx = int(M["m10"] / M["m00"])
    cy = int(M["m01"] / M["m00"])
    cv2.circle(drawing,(cx,cy),1,(255,0,0),3)

    hull = cv2.convexHull(cnt, returnPoints=False)
    defects = cv2.convexityDefects(cnt, hull)

    count_defects = 0
    cv2.drawContours(thresh, contours, -1, (0, 255, 0), 3)

    for i in range(defects.shape[0]):
        s, e, f, d = defects[i, 0]

        start = tuple(cnt[s][0])
        end = tuple(cnt[e][0])
        far = tuple(cnt[f][0])

        a = math.sqrt((end[0] - start[0]) ** 2 + (end[1] - start[1]) ** 2)
        b = math.sqrt((far[0] - start[0]) ** 2 + (far[1] - start[1]) ** 2)
        c = math.sqrt((end[0] - far[0]) ** 2 + (end[1] - far[1]) ** 2)

        angle = math.acos((b ** 2 + c ** 2 - a ** 2) / (2 * b * c)) * 57

        if angle <= 90:
            count_defects += 1
            cv2.circle(crop_img, far, 1, [0, 0, 255], -1)

        cv2.line(crop_img, start, end, [0, 255, 0], 2)

    if count_defects == 1:
        str = "2 Fingers"
        cv2.putText(img, str, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
    elif count_defects == 2:
        str = "3 Fingers"
        cv2.putText(img, str, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
    elif count_defects == 3:
        str = "4 Fingers"
        cv2.putText(img, str, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
    elif count_defects == 4:
        str = "Moving Cursor"
        cv2.putText(img, str, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        if fistflag == 1:
            mouse.release(Button.left)
            fistflag = 0
        mouseLoc = (sx-(cx*sx/camx),cy*sy/camy)
        mouse.position = mouseLoc
        # while mouse.position != mouseLoc:
        #     pass

    else:
        if arearatio < 20:
            str = "Draging Cursor"
            cv2.putText(img, str, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
            if fistflag == 0:
                mouse.press(Button.left)
                fistflag = 1
            mouseLoc = (sx - (cx * sx / camx), cy * sy / camy)
            mouse.position = mouseLoc
            # while mouse.position != mouseLoc:
            #     pass
        else:
            str = "1 Finger"
            cv2.putText(img, str, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

    cv2.imshow("Gesture", img)
    cv2.imshow("Threshold",thresh)
    all_img = np.hstack((drawing, crop_img))
    out.write(img)
    cv2.imshow('Contours', all_img)

    if cv2.waitKey(1) & 0xff == ord('q') or keyboard.is_pressed('q'):
        break

cap.release()
out.release()
cv2.destroyAllWindows()