import cv2
import numpy as np 
import face_recognition

imgElon = face_recognition.load_image_file("images/elon musk 1.jpeg")
imgElon = cv2.cvtColor(imgElon,cv2.COLOR_BGR2RGB)   #converting img to rgb format
imgTest = face_recognition.load_image_file("images/bill gates 1.jpeg")
imgTest = cv2.cvtColor(imgTest,cv2.COLOR_BGR2RGB)

faceLoc = face_recognition.face_locations(imgElon)[0]  #[0] is bcoz of only 1 img
encodeElon = face_recognition.face_encodings(imgElon)[0]
cv2.rectangle(imgElon,(faceLoc[3],faceLoc[0]),(faceLoc[1],faceLoc[2]),(255,0,255),2)   #box creation

faceLocTest = face_recognition.face_locations(imgTest)[0]  #[0] is bcoz of only 1 img
encodeTest = face_recognition.face_encodings(imgTest)[0]
cv2.rectangle(imgTest,(faceLocTest[3],faceLocTest[0]),(faceLocTest[1],faceLocTest[2]),(255,0,255),2)   #box creation

results = face_recognition.compare_faces([encodeElon],encodeTest)
faceDis = face_recognition.face_distance([encodeElon],encodeTest)
print(results,faceDis)

cv2.putText(imgTest,f'{results}{round(faceDis[0],2)}',(50,150),cv2.FONT_HERSHEY_COMPLEX_SMALL,1,(0,0,255),2)   #(50,100) is coordinates of text 
cv2.imshow('Elon musk',imgElon)
cv2.imshow('Elon Test',imgTest)
cv2.waitKey(0) 