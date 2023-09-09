import cv2
from random import randrange

trained_face_data = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
img = cv2.imread('people.jpg')
grayscaled_img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
face_coordinates = trained_face_data.detectMultiScale(grayscaled_img)
for(x,y,w,h) in face_coordinates:
    cv2.rectangle(img, (x, y), (x + w, y + h), (randrange(256), randrange(256), randrange(256)), 5)
# (x,y,w,h) = face_coordinates[6]

h,w,channels = img.shape
neww = 800
newh = int(neww * (h/w))
img = cv2.resize(img,(neww, newh))
# cv2.rectangle(img,(339,410),(339 + 1323,410 + 1323),(0,255,0),2)

cv2.imshow("Face Detector", img)
cv2.waitKey(0)

print(face_coordinates)

print("Code Completed")