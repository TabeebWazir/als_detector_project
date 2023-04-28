import cv2 # 1 camera related import
from cvzone.HandTrackingModule import HandDetector # 2 hand detection import
import numpy as np
import math #6
import time #8

cap = cv2.VideoCapture(0) # 1 code for initializing camera
detector = HandDetector(maxHands=1) # 2 code for detecting hands

offset = 20
imgSize = 300 # 4

folder = "Data/GOODBYE" #8 Store capture gesters in folder A
counter = 0 #8 countes the number of images recored and saved in folder A

while True: # 1 and 2
    sucess, img = cap.read() # 1
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
        else:
            k = imgSize / w # 7
            hCal = math.ceil(k * h) # 7
            imgResize = cv2.resize(imgCrop,(imgSize,hCal)) #6
            imgResizeShape = imgResize.shape # 6
            hGap = math.ceil((imgSize-hCal)/2) # centers the imge in white screen
            imgWhite[hGap:hCal + hGap , :] = imgResize # 7 allows more room for wide hand gestures


        cv2.imshow("ImageCrop", imgCrop) # 5
        cv2.imshow("ImageWhite", imgWhite) # 4

    cv2.imshow("Image", img) # 1
    key = cv2.waitKey(1) # 1
    if key == ord("s"): #8 if hold keyboard key s it will record the gestore and save them in folder A
        counter += 1 #8
        cv2.imwrite(f'{folder}/Image_{time.time()}.jpg',imgWhite) #8
        print(counter) #8

