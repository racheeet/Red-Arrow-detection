import cv2 as cv
import numpy as np
import math as math


def img2mask(img):
    hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)
    lower_red = np.array([0, 70, 50])
    upper_red = np.array([10, 255, 255])

    mask = cv.inRange(hsv, lower_red, upper_red)
    result = cv.bitwise_and(img, img, mask=mask)
    return result

if __name__ == "__main__" :
    capture = cv.VideoCapture(0)
    while True:
        isTrue, img = capture.read()
        # img = get_cntrd(img)
        blank = np.zeros(img.shape, dtype="uint8")

        # b, g, r = cv.split(img)
        # img = cv.merge([b, g, r])

        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        blurr = cv.GaussianBlur(gray, (3, 3), cv.BORDER_DEFAULT)
        ret, thresh = cv.threshold(blurr, 229, 245, cv.THRESH_BINARY)
        # canny = cv.Canny(thresh, 125, 175)
        contours, heirarchies = cv.findContours(
            thresh, cv.RETR_LIST, cv.CHAIN_APPROX_NONE)
        frame = blank
        # cv.drawContours(frame, contours, -1, (0, 0, 255), 2)

        r_areas = [cv.contourArea(c) for c in contours]
        max_rarea = np.max(r_areas)
        for cnt in contours:
            if((cv.contourArea(cnt) > max_rarea * 0.2) and (cv.contourArea(cnt) < 0.9*max_rarea)):
                cv.drawContours(frame, [cnt], -1, (0, 0, 255), 3)

                peri = 0.02 * cv.arcLength(cnt, True)
                approximations = cv.approxPolyDP(cnt, peri, True)
                cv.drawContours(frame, [approximations], 0, (0), -1)
                x, y, w, h = cv.boundingRect(cnt)
                cv.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)

                if len(approximations) == 7:
                    cv.putText(frame, "ARROW",  (frame.shape[0]//2,
                            frame.shape[1]//2),
                            cv.FONT_HERSHEY_COMPLEX, 1.0, (0, 255, 0), 2)

                rows, cols = frame.shape[:2]
                [vx, vy, x, y] = cv.fitLine(cnt, cv.DIST_L2, 0, 0.01, 0.01)
                left_y = int((-x*vy/vx) + y)
                right_y = int(((cols-x)*vy/vx)+y)
                frame = cv.line(frame, (cols-1, right_y),
                                (0, left_y), (0, 255, 0), 2)

                slope = 0
                slope = math.degrees(
                    math.atan2((cols-1), (right_y-left_y)) - math.atan2(1, 0))
                slope = 90-slope
                print(vx, vy, x, y)
                print(slope)

                slope = str(slope)
                cv.putText(frame, slope,  (45, 45),
                        cv.FONT_HERSHEY_COMPLEX, 1.0, (0, 255, 0), 2)

        cv.imshow("real", frame)
        cv.imshow("img", img)

        if cv.waitKey(1) & 0xFF == ord('q'):
            break

    cv.capture.release(capture)
    cv.destroyAllWindows()


        
