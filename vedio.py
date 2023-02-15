import cv2

trainedDataset= cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
video = cv2.VideoCapture('videos/WIN_20221030_22_16_00_Pro.mp4')
while True:
    success, frame= video.read()
    if success==True:
        gray_image= cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces= trainedDataset.detectMultiScale(gray_image)
        for x, y, w, h in faces:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 5)
        cv2.imshow('video', frame)
        cv2.waitKey(1)
    else:
        print("Video completed or frame Nil")
        break