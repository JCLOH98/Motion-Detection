import cv2 as cv

cap = cv.VideoCapture(0);

_,frame1 = cap.read();
mf1 = cv.flip(frame1,1);
_,frame2= cap.read();
mf2 = cv.flip(frame2,1);

def empty(x):
    pass

cv.namedWindow("End product");

cv.createTrackbar("AreaL","End product",0,10000,empty);
cv.createTrackbar("AreaH","End product",10000,10000,empty);

while (cap.isOpened()):
    areaL = cv.getTrackbarPos("AreaL","End product");
    areaH = cv.getTrackbarPos("AreaH","End product");
    mirrorframe = cv.flip(frame2,1);

    #the difference between 1st frame and 2nd frame
    diff = cv.absdiff(frame1,frame2);
    mirrordiff = cv.flip(diff,1);
    cvtColordiff = cv.cvtColor(mirrordiff,cv.COLOR_BGR2GRAY);
    #(ksizewidth,ksizeheight) needs to be odd number
    blurdiff = cv.GaussianBlur(cvtColordiff,(3,3),0);
    _,thresdiff = cv.threshold(blurdiff,30,255,cv.THRESH_BINARY);
    dilateddiff = cv.dilate(thresdiff, None, iterations=3);
    contours, _ = cv.findContours(dilateddiff, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE);
    cv.imshow("Diff",dilateddiff);

    #end product
    for c in contours:
                (x,y,w,h) = cv.boundingRect(c);

                if (cv.contourArea(c) >= areaL and cv.contourArea(c) <= areaH):
                    #img,pt1,pt2,color,thickness,lineType
                    cv.rectangle(mirrorframe,(x,y),(x+w,y+h),(255,0,0),2);
    cv.imshow("End product",mirrorframe);
    if (cv.waitKey(1) == ord('q')):
        break;
    
    frame1 = frame2;
    _,frame2 = cap.read();
cv.destroyAllWindows();
cap.release();
