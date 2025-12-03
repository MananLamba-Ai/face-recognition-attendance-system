import cv2
import os
import numpy as np

dataset_path = "dataset"
recognizer = cv2.face.LBPHFaceRecognizer_create()

faces = []
ids = []

# Load images and labels
for file in os.listdir(dataset_path):
    img_path = os.path.join(dataset_path, file)
    gray_img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

    id = int(file.split(".")[1])
    faces.append(gray_img)
    ids.append(id)

ids = np.array(ids)

# Train model
recognizer.train(faces, ids)
recognizer.write("trainer.yml")

print("Training completed. Model saved as trainer.yml")
