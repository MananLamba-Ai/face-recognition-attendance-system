import cv2
import os

# Create dataset directory if not exists
dataset_path = "dataset"
if not os.path.exists(dataset_path):
    os.makedirs(dataset_path)

# Take user ID input
user_id = input("Enter User ID: ")

# Load Haar Cascade
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

cap = cv2.VideoCapture(0)
count = 0

while True:
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        count += 1
        filename = f"{dataset_path}/User.{user_id}.{count}.jpg"
        cv2.imwrite(filename, gray[y:y+h, x:x+w])
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

    cv2.imshow("Capturing Dataset", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    if count >= 50:
        break

cap.release()
cv2.destroyAllWindows()

print("Dataset collection completed.")
