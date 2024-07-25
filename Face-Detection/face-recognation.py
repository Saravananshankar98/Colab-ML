import cv2 as cv

name = ["Unknown", "Saravanan", "Dinesh"]

def face_reg():
    face_classifier = cv.CascadeClassifier('haarcascade_frontalface_default.xml')
    
    if face_classifier.empty():
        print("Error: Could not load face classifier.")
        return

    cap = cv.VideoCapture(0)

    if not cap.isOpened():
        print("Error: Could not open video capture.")
        return

    model = cv.face.LBPHFaceRecognizer_create()
    model.read('recognizer/trainingData.xml')

    confidence_threshold = 100

    while True:
        ret, frame = cap.read()

        if not ret:
            print("Error: Could not read frame.")
            break

        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        faces = face_classifier.detectMultiScale(gray, 1.3, 5)

        for (x, y, w, h) in faces:
            cv.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

            offset = 10
            x_start = max(0, x - offset)
            y_start = max(0, y - offset)
            x_end = min(frame.shape[1], x + w + offset)
            y_end = min(frame.shape[0], y + h + offset)

            face_section = gray[y_start:y_end, x_start:x_end]
            face_section = cv.resize(face_section, (100, 100))

            result = model.predict(face_section)

            if result[1] < confidence_threshold:
                display_string = name[result[0]]
            else:
                display_string = 'No match'

            (text_width, text_height), _ = cv.getTextSize(display_string, cv.FONT_HERSHEY_COMPLEX, 1, 2)
            text_x = x + (w - text_width) // 2
            text_y = y - 10 if y - 10 > 10 else y + h + text_height + 10

            cv.putText(frame, display_string, (text_x, text_y), cv.FONT_HERSHEY_COMPLEX, 1, (255, 0, 0), 2)

        cv.imshow('Face Cropper', frame)

        if cv.waitKey(1) == 13:
            break

    cap.release()
    cv.destroyAllWindows()

face_reg()