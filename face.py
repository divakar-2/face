import cv2

#Trained DataSet
trainedDataset= cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

#Read a Image
img = cv2.imread('images/WIN_20221030_21_01_42_Pro.jpg')

#Convert into grayscale
gray=cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
faces = trainedDataset.detectMultiScale(gray)
print(faces)
for x, y, w, h in faces:
    cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 5)
cv2.imshow('Diva', img)
cv2.imshow('Gray', gray)
cv2.waitKey()