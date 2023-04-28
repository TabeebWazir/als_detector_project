import cv2 # 1 camera related import
from cvzone.HandTrackingModule import HandDetector # 2 hand detection import
from cvzone.ClassificationModule import Classifier # 9
import numpy as np
import math #6


cap = cv2.VideoCapture(0) # 1 code for initializing camera
detector = HandDetector(maxHands=1) # 2 code for detecting hands
classifier = Classifier("Model/keras_model.h5", "Model/labels.txt") # 9

offset = 20
imgSize = 500 # 4

folder = "Data/C" #8 Store capture gesters in folder A
counter = 0 #8 countes the number of images recored and saved in folder A

labels = ["GOODBYE","HELLO","I LOVE YOU","NO","SORRY","THANK YOU","WELCOME","YES"] #9

while True: # 1 and 2
    sucess, img = cap.read() # 1
    imgOutput = img.copy()
    hands, img = detector.findHands(img) # 2 Detects left and right hands
    if hands: # 3 shows the hands only on separate display
        hand = hands[0]
        x,y,w,h = hand['bbox']

        imgWhite = np.ones((imgSize,imgSize,3),np.uint8)*255 # 4 displays white screen

        imgCrop = img[y-offset:y + h+offset, x:x + w+offset] # 3

        imgCropShape = imgCrop.shape # 5

        # imgWhite[0:imgCropShape[0], 0:imgCropShape[1]] = imgCrop # 5 closes displays as it reaches the width of the white screen

        aspectRatio = h / w # 6 resize hand capture screen to be compitable with white screen

        if aspectRatio >1: #6
            k = imgSize / h #6
            wCal = math.ceil(k * w) #6
            imgResize = cv2.resize(imgCrop,(wCal,imgSize)) #6
            imgResizeShape = imgResize.shape # 6
            wGap = math.ceil((imgSize-wCal)/2) # centers the imge in white screen
            imgWhite[:, wGap:wCal+wGap] = imgResize # 6
            prediction, index = classifier.getPrediction(imgWhite,draw=False) #11
            print(prediction, index) #9

        else:
            k = imgSize / w # 7
            hCal = math.ceil(k * h) # 7
            imgResize = cv2.resize(imgCrop,(imgSize,hCal)) #6
            imgResizeShape = imgResize.shape # 6
            hGap = math.ceil((imgSize-hCal)/2) # centers the imge in white screen
            imgWhite[hGap:hCal + hGap , :] = imgResize # 7 allows more room for wide hand gestures
            prediction, index = classifier.getPrediction(imgWhite,draw=False) #11
        cv2.rectangle(imgOutput, (x - offset, y - offset-70),
                      (x - offset+400, y - offset-50+70), (0, 150, 0), cv2.FILLED)  # 12 a filled pink box for text
        cv2.putText(imgOutput,labels[index],(x,y-26),cv2.FONT_HERSHEY_COMPLEX,2,(255,255,255),2) #10
        cv2.rectangle(imgOutput,(x-offset,y-offset),(x+w+offset,y+h+offset),(0,150,0),4) #11 adds a pink border around the images



        cv2.imshow("ImageCrop", imgCrop) # 5
        cv2.imshow("ImageWhite", imgWhite) # 4

    cv2.imshow("Image", imgOutput) # 9
    cv2.waitKey(1) # 1


