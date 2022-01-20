from multiprocessing.connection import wait
from tkinter import W
import cv2 as cv 
import mediapipe as mp
import time 

# My First Project of Computer Vision - Hand Detecter

cap = cv.VideoCapture(0) # It can be 0,1,2,3 depending how much camera you have - In general 0

mpHands = mp.solutions.hands
hands = mpHands.Hands()
mpDraw = mp.solutions.drawing_utils

pTime = 0
cTime = 0

while True :
    sucess, img = cap.read(0)
    imgRGB = cv.cvtColor(img, cv.COLOR_BGR2RGB)
    results = hands.process(imgRGB)
    # print(results.multi_hand_landmarks)

    if results.multi_hand_landmarks:                    # If script detect hand is true : 
        for handLms in results.multi_hand_landmarks:        #  For all hands detected :
            # for id, lm in enumerate(handLms.landmark):
            #     h, w, c = img.shape                       
            #     cx, cy = int(lm.x*w), int(lm.y*h) 

            mpDraw.draw_landmarks(img, handLms, mpHands.HAND_CONNECTIONS) # Draw all 20 points of the hand + connexion line


    cTime = time.time()
    fps = 1/(cTime-pTime)
    pTime = cTime

    cv.putText(img,str(int(fps)), (10,70), cv.FONT_HERSHEY_PLAIN, 3, (255,0,255), 3) # Show FPS On Screen

    cv.imshow('Image', img) 
    cv.waitKey(0)

    #WhiteRoseX
