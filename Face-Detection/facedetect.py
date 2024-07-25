import cv2 as cv
import ctypes

user32 = ctypes.windll.user32
screen_width = user32.GetSystemMetrics(0)
screen_height = user32.GetSystemMetrics(1)

webcam = cv.VideoCapture('data/videos/peoples.mp4')

face_cascade_path = 'haarcascade_frontalface_default.xml'
face_cascade = cv.CascadeClassifier(face_cascade_path)

cv.namedWindow('webcam', cv.WINDOW_FULLSCREEN)

while True:
    successful_frame_read, frame = webcam.read()
    if not successful_frame_read:
        break
    
    blackWhite = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(blackWhite, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    
    for (x, y, w, h) in faces:
        cv.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
    
    cv.putText(frame, f'Faces detected: {len(faces)}', (10, 30), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    
    frame_resized = cv.resize(frame, (screen_width, screen_height))
    
    cv.imshow('webcam', frame_resized)
    
    key = cv.waitKey(1)
    if key == 27:
        break

webcam.release()
cv.destroyAllWindows()
