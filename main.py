import cv2
import matplotlib.pyplot as plt

photo_name=input("Enter photo name with .jpeg: ")
photo = cv2.imread(photo_name)

face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
eye_cascade = cv2.CascadeClassifier("haarcascade_eye.xml")

faces = face_cascade.detectMultiScale(photo, scaleFactor=1.075, minNeighbors=5, minSize=(15,15))

for(x,y,w,h) in faces:
  cv2.rectangle(photo, (x,y),(x+w,y+h), (255,0,0),2 )
  roi = photo[y:y+h, x:x+w]
  gozler = eye_cascade.detectMultiScale(roi)
  for(ex,ey,ew,eh) in gozler:
    cv2.rectangle(roi, (ex,ey), (ex+ew, ey+eh), (0,255,0),1)

print("Number Of Faces = ",format(len(faces)))

plt.figure(figsize=(8,12))
plt.imshow(photo)
plt.show()
