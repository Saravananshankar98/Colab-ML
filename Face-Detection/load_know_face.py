import face_recognition
import os

def load_known_faces(known_faces_dir):
    known_face_encodings = []
    known_face_names = []

    # Iterate over each directory in the known_faces directory
    for person_name in os.listdir(known_faces_dir):
        person_dir = os.path.join(known_faces_dir, person_name)
        if os.path.isdir(person_dir):
            # Iterate over each image in the person's directory
            for image_name in os.listdir(person_dir):
                image_path = os.path.join(person_dir, image_name)
                image = face_recognition.load_image_file(image_path)
                # Encode the face(s) in the image
                face_encodings = face_recognition.face_encodings(image)
                
                for face_encoding in face_encodings:
                    known_face_encodings.append(face_encoding)
                    known_face_names.append(person_name)

    return known_face_encodings, known_face_names

# Example usage
known_faces_dir = 'known_faces'
known_face_encodings, known_face_names = load_known_faces(known_faces_dir)
