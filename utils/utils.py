import cv2
import numpy as np

def post_process(img):    

    img = np.array(img, dtype=np.uint8) * 255

    gray = cv2.GaussianBlur(img, (5, 5), 0)
    _, binary = cv2.threshold(gray, 50, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(binary, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    
    area = []
    for j in range(len(contours)):
        area.append(cv2.contourArea(contours[j]))

    if len(area) == 0:
        return img
    else:
        max_idx = np.argmax(area)
        for k in range(len(contours)):
            if k != max_idx:
                cv2.fillPoly(binary, [contours[k]], 0)
        return binary