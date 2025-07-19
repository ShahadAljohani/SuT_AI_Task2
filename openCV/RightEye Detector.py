

import cv2
import time

#Right eye face classifier
rightEye_classifier = cv2.CascadeClassifier('haarcascade_righteye_2splits.xml')

cap = cv2.VideoCapture('owl2.mp4')

while cap.isOpened():
    
    time.sleep(.05) #delay for aaproximately 20FPS = more human-friendly 

    ret, frame = cap.read()

    if not ret: #if the video has ended and no frame is returned the program will stop
        break

    frame = cv2.resize(frame, (640, 480)) #reduce the video size 

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) #B2GRAY is for haar cascade classifiers, because it works best on them
    #what's Grayscale? it is a method of showing images using only a range of shades from black to white instead of colors

    eyes = rightEye_classifier.detectMultiScale(gray, 1.4, 2)

    #Extract bounding boxes for any bodies identified 
    for (x,y,w,h) in eyes:
        padding = 7 #to add extra space inside the rectangle to make it visually larger
        cv2.rectangle(frame, (x - padding, y - padding), (x+w+padding, y+h+padding), (255, 0, 0), 2) #used the padding varaibles to adjust rectangle size

    cv2.imshow('Owl Right Eye Detection', frame)

    if cv2.waitKey(1)== 13: #is the Enter Key
        break
        
cap.release()
cv2.destroyAllWindows()