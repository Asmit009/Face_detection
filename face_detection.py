import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv

img = cv.imread('lady1.jpg')
cv.imshow('lady',img)

gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
cv.imshow('lady',gray)

haar_cascade = cv.CascadeClassifier('haar_cascade.xml')

faces_rec = haar_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=2)

print(f'number of faces = {len(faces_rec)}')

for (x,y,w,h) in faces_rec:
    rectangle = cv.rectangle(img,(x,y),(x+w,y+h),(0,255,0), thickness=2)

cv.imshow('img',rectangle)
cv.waitKey(0)