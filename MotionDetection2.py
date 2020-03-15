import cv2 as cv
import numpy as np

cap = cv.VideoCapture(0);
ret, frame = cap.read();
avgframe = np.float32(frame);


opening_kernel = np.ones((13,13),np.uint8);

cv.namedWindow("End product");
while True:
    ret,frame = cap.read();
    # real time frame
    #cv.imshow("Ori frame",frame);
    #the smaller the alpha, the slower the better
    cv.accumulateWeighted(frame,avgframe,0.25);
    resframe = cv.convertScaleAbs(avgframe);
    #calculate background from the real time frame
    #cv.imshow("Background frame", resframe);

    #seperate background and foreground
    difframe = cv.absdiff(frame,resframe);
    #cv.imshow("Diff frame",difframe);

    #convert the abs diff image into greyscale
    greydifframe = cv.cvtColor(difframe,cv.COLOR_BGR2GRAY);
    #cv.imshow("Gray diff frame",greydifframe);
    
    #(ksizewidth,ksizeheight) needs to be odd number,
    #the bigger the value, the blurrer it is
    blurdiff = cv.GaussianBlur(greydifframe,(21,21),0);
    #cv.imshow("Blur diff frame",blurdiff);
    _,thresdiff = cv.threshold(blurdiff,15,255,cv.THRESH_BINARY);
    dilateddiff = cv.dilate(thresdiff, None, iterations=20);
    openeddiff = cv.morphologyEx(thresdiff,cv.MORPH_OPEN,opening_kernel);
    _, contours, _ = cv.findContours(dilateddiff, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE);
    #contours, _ = cv.findContours(dilateddiff, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE);
    cv.imshow("After procs Diff",dilateddiff);
    #cv.drawContours(frame, contours, -1, (0,255,0), 2);
    

    #end product
    #'''
    rect_cont = [];
    for c in contours:
        rect_cont.append(cv.contourArea(c));
    if (len(rect_cont) != 0 and len(rect_cont) >= 5):
        rect_cont.sort();
        mid_area = rect_cont[int(len(rect_cont)*2/3)];
    else:
        mid_area = 0;
    for c in contours:
                (x,y,w,h) = cv.boundingRect(c);
                if (cv.contourArea(c) >= mid_area):
                    cv.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2);
    #'''
    cv.imshow("End product",frame);
    
    key = cv.waitKey(1);
    if key==ord("q"):
        break

cv.destroyAllWindows();
cap.release();
