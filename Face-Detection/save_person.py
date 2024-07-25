import cv2 as cv
import face_recognition
import os

# Load known faces
known_face_encodings = []
known_face_names = []

known_faces_dir = 'known_faces'

for filename in os.listdir(known_faces_dir):
    if filename.endswith(".jpg") or filename.endswith(".png"):
        image_path = os.path.join(known_faces_dir, filename)
        image = face_recognition.load_image_file(image_path)
        face_encoding = face_recognition.face_encodings(image)[0]
        known_face_encodings.append(face_encoding)
        known_face_names.append(os.path.splitext(filename)[0])

# Initialize webcam
webcam = cv.VideoCapture(0)

while True:
    successful_frame_read, frame = webcam.read()
    rgb_frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
    
    # Find all the faces and face encodings in the current frame
    face_locations = face_recognition.face_locations(rgb_frame)
    face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)
    
    face_names = []
    for face_encoding in face_encodings:
        # Check if the face is a match for the known face(s)
        matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
        name = "Unknown"
        
        # Use the known face with the smallest distance to the new face
        face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
        best_match_index = min(range(len(face_distances)), key=lambda i: face_distances[i])
        
        if matches[best_match_index]:
            name = known_face_names[best_match_index]
        
        face_names.append(name)
    
    # Display the results
    for (top, right, bottom, left), name in zip(face_locations, face_names):
        # Draw a box around the face
        cv.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
        
        # Draw a label with a name below the face
        cv.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 255, 0), cv.FILLED)
        font = cv.FONT_HERSHEY_SIMPLEX
        cv.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)
    
    # Display the resulting image
    cv.imshow('webcam', frame)
    
    # Hit 'q' on the keyboard to quit!
    if cv.waitKey(1) & 0xFF == ord('q'):
        break

# Release handle to the webcam
webcam.release()
cv.destroyAllWindows()
