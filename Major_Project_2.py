import numpy as np
import cv2
import glob

criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
#object points
objp = np.zeros((6*9,3), np.float32)
objp[:,:2] = np.mgrid[0:9,0:6].T.reshape(-1,2)
# Arrays to store object points and image points from all the images.
objpoints = []
imgpoints = []

#video = cv2.VideoCapture("around.mp4")
video = cv2.VideoCapture(0)
if not video.isOpened():
    print ("Could not open video")
    sys.exit()


ok, frame = video.read()
if not ok:
    print ('Cannot read video file')
    sys.exit()
#for fname in images:
counter = -1;
while True:
    cv2.waitKey(1)
    counter = counter+1;
    if counter%10 ==0:
        objpoints = []
        imgpoints = []
        ok, img = video.read()
        if not ok:
            break
        #print(fname)
        if(ok):
            cont = False;
        #fname = "C:/Users/Srikanth/Documents/Robo/CameraCalib/c4s.jpg"
        #img = cv2.imread(fname)

            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            ret,thresh = cv2.threshold(gray,127,255,0)
            thresh2 = cv2.adaptiveThreshold(thresh,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,\
                cv2.THRESH_BINARY,5,10)
            blur = cv2.GaussianBlur(thresh,(5,5),0)
    # Find the chess board corners
            ret, corners = cv2.findChessboardCorners(thresh, (9,6),None)
    # If found, add object points, image points
            if ret == True:
                objpoints.append(objp)
                corners2=cv2.cornerSubPix(thresh,corners, (11,11), (-1,-1), criteria)
                imgpoints.append(corners)
        # Draw and display the corners
                cv2.drawChessboardCorners(thresh, (9,6), corners2, ret,)
                cv2.imshow('img', thresh)
                cv2.waitKey(10)

                ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

                rotVec = np.zeros((3, 3), np.float32)
                rotVec,_=cv2.Rodrigues(np.array(rvecs))


                rotf = -rotVec.T
                cameraPosition =  np.dot(rotf,np.array(tvecs))
                print(cameraPosition)

            else:
            #cv2.imshow('img', blur)
            #cv2.waitKey(5000)
                cv2.imshow('img', thresh)
        #break
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

cv2.destroyAllWindows()

#print(objpoints[0])
# ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
#
# rotVec = np.zeros((3, 3), np.float32)
# rotVec,_=cv2.Rodrigues(np.array(rvecs))
#
#
# rotf = -rotVec.T
# cameraPosition =  np.dot(rotf,np.array(tvecs))
# print(cameraPosition)
