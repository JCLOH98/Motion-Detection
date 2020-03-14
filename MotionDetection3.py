import cv2 as cv
import numpy as np

cap = cv.VideoCapture(0);
ret, frame = cap.read();
backSubMOG = cv.createBackgroundSubtractorMOG2();
backSubKNN = cv.createBackgroundSubtractorKNN();

while True:
    ret, frame = cap.read();
    cv.imshow("Ori frame",frame);
    fgMaskMOG = backSubMOG.apply(frame);
    fgMaskKNN = backSubKNN.apply(frame);
    cv.imshow("MOG fg mask",fgMaskMOG);
    cv.imshow("KNN fg mask",fgMaskKNN);
    
    key = cv.waitKey(1);
    if key==ord("q"):
        break

cv.destroyAllWindows();
cap.release();
