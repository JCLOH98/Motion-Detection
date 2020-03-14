import cv2 as cv
import numpy as np

cap = cv.VideoCapture(0);
ret, frame = cap.read();
avgframe = np.float32(frame);

def empty(x):
    pass

cv.namedWindow("End product");
cv.createTrackbar("AreaL","End product",0,10000,empty);
cv.createTrackbar("AreaH","End product",10000,10000,empty);

while True:
    
    areaL = cv.getTrackbarPos("AreaL","End product");
    areaH = cv.getTrackbarPos("AreaH","End product");
    ret,frame = cap.read();
    # real time frame
    #cv.imshow("Ori frame",frame);
    #the smaller the alpha, the slower the better
    cv.accumulateWeighted(frame,avgframe,0.3);
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
    blurdiff = cv.GaussianBlur(greydifframe,(13,13),0);
    #cv.imshow("Blur diff frame",blurdiff);
    _,thresdiff = cv.threshold(blurdiff,15,255,cv.THRESH_BINARY);
    dilateddiff = cv.dilate(thresdiff, None, iterations=2);
    #_, contours, _ = cv.findContours(dilateddiff, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE);
    contours, _ = cv.findContours(dilateddiff, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE);
    #cv.imshow("Diff",dilateddiff);
    #cv.drawContours(frame, contours, -1, (0,255,0), 2);
    

    #end product
    for c in contours:
                (x,y,w,h) = cv.boundingRect(c);

                if (cv.contourArea(c) >= areaL and cv.contourArea(c) <= areaH):
                    #img,pt1,pt2,color,thickness,lineType
                    cv.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2);
    cv.imshow("End product",frame);
    
    key = cv.waitKey(1);
    if key==ord("q"):
        break

cv.destroyAllWindows();
cap.release();
