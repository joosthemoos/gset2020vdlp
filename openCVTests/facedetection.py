import cv2
import numpy as np
import matplotlib as plt
import math
capture = cv2.VideoCapture(0, cv2.CAP_DSHOW)
while (True):
    try:
        ret, frame = capture.read()
        frame = cv2.flip(frame, 1)
        hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        target = hsv_frame[100:500, 100:300]
        # Load the cascade
        face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
        # Convert into grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # Detect faces
        faces = face_cascade.detectMultiScale(gray, 1.1, 5)
        # print(faces[0][1])
        # Draw rectangle around the faces
        #cv2.rectangle(img,(faces[0][0],faces[0][1]),(faces[0][0]+faces[0][2],faces[0][1]+faces[0][3]), (255, 0, 0), 2)
        for (x, y, w, h) in faces:
            print(len(faces))
            if len(faces) == 1:
                print("first", x, y, w, h)
                cv2.rectangle(frame, (x+25, y+25), (x + w-25, y + h), (255, 0, 0), 2)

        crop_img = hsv_frame[y + 25:y + w, x + 25:x + h-25]
        roi = crop_img

        hue, saturation, value = cv2.split(roi)

        hand_hist = cv2.calcHist([roi], [0, 1], None, [180, 256], [0, 180, 0, 256]) #only need hue and sat. [0,1] not value
        dst = cv2.calcBackProject([target], [0, 1], hand_hist, [0, 180, 0, 256], 1)
        disc = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
        disc = cv2.GaussianBlur(disc, (5, 5), 100)
        cv2.filter2D(dst, -1, disc, dst)
        ret, thresh = cv2.threshold(dst, 150, 255, cv2.THRESH_BINARY)
        thresh = cv2.dilate(thresh, None, iterations=5)
        thresh = cv2.merge((thresh, thresh, thresh))
        hist_mask_image = cv2.bitwise_and(target, thresh)
        hist_mask_image = cv2.erode(hist_mask_image, None, iterations=4)
        hist_mask_image = cv2.dilate(hist_mask_image, None, iterations=4)
        gray_hist_mask_image = cv2.cvtColor(hist_mask_image, cv2.COLOR_BGR2GRAY)
        ret, thresh = cv2.threshold(gray_hist_mask_image, 0, 255, 0)
        _, cont, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        contour_list = cont
        max_cont = max(contour_list, key=lambda x: cv2.contourArea(x))

        epsilon = 0.0005 * cv2.arcLength(max_cont, True)
        approx = cv2.approxPolyDP(max_cont, epsilon, True)
        hull = cv2.convexHull(approx, returnPoints=False)
        #areaHand = cv2.contourArea(max_cont)
        #areahull = cv2.contourArea(hull)
        #arearatio = ((areahull - areaHand) / areaHand) * 100
        defects = cv2.convexityDefects(approx, hull)
        defect = 0
        for i in range(defects.shape[0]):
            s, e, f, d = defects[i, 0]
            start = tuple(approx[s][0])
            end = tuple(approx[e][0])
            far = tuple(approx[f][0])
            print(start, end, far)
            # find length of all sides of triangle
            # pythagorean theorem
            a = math.sqrt((end[0] - start[0]) ** 2 + (end[1] - start[1]) ** 2)
            b = math.sqrt((far[0] - start[0]) ** 2 + (far[1] - start[1]) ** 2)
            c = math.sqrt((end[0] - far[0]) ** 2 + (end[1] - far[1]) ** 2)
            # heron's formula to find approx area of hand?
            s = (a + b + c) / 2
            ar = math.sqrt(s * (s - a) * (s - b) * (s - c))
            d = (2 * ar) / a
            angle = math.acos((b ** 2 + c ** 2 - a ** 2) / (2 * b * c)) * 57
            if angle <= 90 and d > 30:
                defect += 1
                cv2.circle(target, far, 3, [255, 0, 0], -1)

            cv2.line(target, start, end, [0, 255, 0], 2)

        defect += 1
        font = cv2.FONT_HERSHEY_SIMPLEX
        if defect == 1:
            #if areaHand < 2000:
                #cv2.putText(frame, 'Put hand in the box', (0, 50), font, 2, (0, 0, 255), 3, cv2.LINE_AA)
            #else:
                #if arearatio < 12:
                 #   cv2.putText(frame, '0', (0, 50), font, 2, (0, 0, 255), 3, cv2.LINE_AA)

                #else:
            cv2.putText(frame, '1', (0, 50), font, 2, (0, 0, 255), 3, cv2.LINE_AA)

        elif defect == 2:
            cv2.putText(frame, '2', (0, 50), font, 2, (0, 0, 255), 3, cv2.LINE_AA)

        elif defect == 3:
            cv2.putText(frame, '3', (0, 50), font, 2, (0, 0, 255), 3, cv2.LINE_AA)

        elif defect == 4:
            cv2.putText(frame, '4', (0, 50), font, 2, (0, 0, 255), 3, cv2.LINE_AA)

        elif defect == 5:
            cv2.putText(frame, '5', (0, 50), font, 2, (0, 0, 255), 3, cv2.LINE_AA)

        elif defect == 6:
            cv2.putText(frame, 'reposition', (0, 50), font, 2, (0, 0, 255), 3, cv2.LINE_AA)

        else:
            cv2.putText(frame, 'reposition', (10, 50), font, 2, (0, 0, 255), 3, cv2.LINE_AA)

        #areahull = cv2.contourArea(hull)
        #areahand = cv2.contourArea(max_cont)
        #arearatio = ((areahull - areahand) / areahand) * 100                           )
        # Display the output '''
        cv2.imshow('frame', frame)
        cv2.imshow('mask', hist_mask_image)
        cv2.imshow('target', target)

    except:
        pass
    k = cv2.waitKey(5) & 0xFF
    if k == 27:
        break

capture.release()
cv2.destroyAllWindows()
