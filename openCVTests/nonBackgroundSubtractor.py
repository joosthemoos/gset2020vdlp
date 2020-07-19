import cv2
import numpy as np
import math

cap = cv2.VideoCapture(0)

while (cap.isOpened()):

    try:  # an error comes if it does not find anything in window as it cannot find contour of max area
        # therefore this try error statement

        ret, frame = cap.read()
        #user camera
        frame = cv2.flip(frame, 1) # flips image frames into right orientation

        #detection camera
        #creates a 3x3 matrix full of 1's (integer) for a mean blur
        kernel = np.ones((3, 3), np.uint8) # noise detection frame

        # define region of interest (where the hand is) within frame
        roi = frame[100:500, 100:300]

        #draws rectangle in frame window
        cv2.rectangle(frame, (100, 100), (300, 400), (0, 255, 0), 0)
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV) # changes color scale from BGR to HSV for readability within target area of frame

        # define skin color range in HSV
        lower_skin = np.array([0, 20, 70], dtype=np.uint8)
        upper_skin = np.array([46, 255, 255], dtype=np.uint8)

        # creates a mask of colors within color ange
        mask = cv2.inRange(hsv, lower_skin, upper_skin)

        # extrapolate the hand to fill dark spots within, interations signify how many erode layers are applied to the mask (the more, the thinner the image)
        mask = cv2.erode(mask, kernel, iterations=5)

        # blur the image - Gaussian blur uses standard deviation methods to give greater weight to central (mean) values within matrix
        mask = cv2.GaussianBlur(mask, (5, 5), 70)

        # find contours
        _, contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        # find contour of max area(hand) and disregards noise
        handContour = max(contours, key=lambda x: cv2.contourArea(x))

        # approx the contour a little
        epsilon = 0.0005 * cv2.arcLength(handContour, True)
        approx = cv2.approxPolyDP(handContour, epsilon, True)

        # make convex hull around hand
        handOutline = cv2.convexHull(handContour)

        # define area of hull and area of hand
        areahull = cv2.contourArea(handOutline)
        areaHand = cv2.contourArea(handContour)

        # find the percentage of area not covered by hand in convex hull
        arearatio = ((areahull - areaHand) / areaHand) * 100

        # find the defects in convex hull with respect to hand
        handOutline = cv2.convexHull(approx, returnPoints=False)
        defects = cv2.convexityDefects(approx, handOutline)

        # no. of defects
        defect = 0

        # code for finding no. of defects due to fingers
        for i in range(defects.shape[0]):
            s, e, f, d = defects[i, 0]
            start = tuple(approx[s][0])
            end = tuple(approx[e][0])
            far = tuple(approx[f][0])
            pt = (100, 180)

            # find length of all sides of triangle
            a = math.sqrt((end[0] - start[0]) ** 2 + (end[1] - start[1]) ** 2)
            b = math.sqrt((far[0] - start[0]) ** 2 + (far[1] - start[1]) ** 2)
            c = math.sqrt((end[0] - far[0]) ** 2 + (end[1] - far[1]) ** 2)
            s = (a + b + c) / 2
            ar = math.sqrt(s * (s - a) * (s - b) * (s - c))

            # distance between point and convex hull
            d = (2 * ar) / a

            # apply cosine rule here
            angle = math.acos((b ** 2 + c ** 2 - a ** 2) / (2 * b * c)) * 57

            # ignore angles > 90 and ignore points very close to convex hull(they generally come due to noise)
            if angle <= 90 and d > 30:
                defect += 1
                cv2.circle(roi, far, 3, [255, 0, 0], -1)

            # draw lines around hand
            cv2.line(roi, start, end, [0, 255, 0], 2)

        defect += 1

        # print corresponding gestures which are in their ranges
        font = cv2.FONT_HERSHEY_SIMPLEX
        if defect == 1:
            if areaHand < 2000:
                cv2.putText(frame, 'Put hand in the box', (0, 50), font, 2, (0, 0, 255), 3, cv2.LINE_AA)
            else:
                if arearatio < 12:
                    cv2.putText(frame, '0', (0, 50), font, 2, (0, 0, 255), 3, cv2.LINE_AA)
                #elif arearatio < 17.5:
                   # cv2.putText(frame, 'Best of luck', (0, 50), font, 2, (0, 0, 255), 3, cv2.LINE_AA)

                else:
                    cv2.putText(frame, '1', (0, 50), font, 2, (0, 0, 255), 3, cv2.LINE_AA)

        elif defect == 2:
            cv2.putText(frame, '2', (0, 50), font, 2, (0, 0, 255), 3, cv2.LINE_AA)

        elif defect == 3:

            #if arearatio < 27:
            cv2.putText(frame, '3', (0, 50), font, 2, (0, 0, 255), 3, cv2.LINE_AA)
            #else:
                #cv2.putText(frame, 'ok', (0, 50), font, 2, (0, 0, 255), 3, cv2.LINE_AA)

        elif defect == 4:
            cv2.putText(frame, '4', (0, 50), font, 2, (0, 0, 255), 3, cv2.LINE_AA)

        elif defect == 5:
            cv2.putText(frame, '5', (0, 50), font, 2, (0, 0, 255), 3, cv2.LINE_AA)

        elif defect == 6:
            cv2.putText(frame, 'reposition', (0, 50), font, 2, (0, 0, 255), 3, cv2.LINE_AA)

        else:
            cv2.putText(frame, 'reposition', (10, 50), font, 2, (0, 0, 255), 3, cv2.LINE_AA)

        # show the windows
        cv2.imshow('mask', mask)
        cv2.imshow('frame', frame)
    except:

        pass

    k = cv2.waitKey(5) & 0xFF
    if k == 27:
        break
print("not working")
cv2.destroyAllWindows()
cap.release()
