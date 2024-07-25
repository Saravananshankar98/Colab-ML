

import cv2 as cv
import numpy as np
from PIL import Image
import os

def train_classifier():
    data_dir = "data"
    paths = [os.path.join(data_dir, f) for f in os.listdir(data_dir) if os.path.isfile(os.path.join(data_dir, f))]

    faces = []
    ids = []

    for image in paths:
        try:
            img = Image.open(image).convert('L')  # Convert image to grayscale
            imageNp = np.array(img, 'uint8')
            id = int(os.path.split(image)[1].split('.')[1])

            faces.append(imageNp)
            ids.append(id)
            cv.imshow("Training", imageNp)
            cv.waitKey(1)  # Wait for a short time
        except Exception as e:
            print(f"Error processing image {image}: {e}")

    if len(faces) == 0:
        print("No faces found. Training cannot proceed.")
        return

    ids = np.array(ids)

    try:
        # Ensure the directory exists
        if not os.path.exists('recognizer'):
            os.makedirs('recognizer')

        # Train the classifier
        recognizer = cv.face.LBPHFaceRecognizer_create()
        recognizer.train(faces, ids)
        recognizer.save("recognizer/trainingData.xml")
        print("Training Done!!")
    except cv.error as e:
        print(f"OpenCV error during training: {e}")
    except Exception as e:
        print(f"Unexpected error during training: {e}")
    finally:
        cv.destroyAllWindows()

# Call the training function
train_classifier()

