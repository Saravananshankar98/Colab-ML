import cv2 as cv
import os

if not os.path.exists('data'):
    os.makedirs('data')

face_classifier = cv.CascadeClassifier('haarcascade_frontalface_default.xml')

def face_cropped(img):
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    faces = face_classifier.detectMultiScale(gray, 1.3, 5)
    if len(faces) == 0:
        return None
    for (x, y, w, h) in faces:
        cropped_face = img[y:y+h, x:x+w]
    return cropped_face

cap = cv.VideoCapture(0)
img_id = 0
id = 1

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        break

    if face_cropped(frame) is not None:
        img_id += 1
        face = cv.resize(face_cropped(frame), (300, 300))
        face = cv.cvtColor(face, cv.COLOR_BGR2GRAY)
        file_name_path = f'data/user.{id}.{img_id}.jpg'
        cv.imwrite(file_name_path, face)
        cv.putText(face, str(img_id), (50, 50), cv.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)
        cv.imshow('Face Cropper', face)

    if cv.waitKey(1) == 13 or img_id == 100:
        break

cap.release()
cv.destroyAllWindows()
print("Collecting Samples Complete!!!")

