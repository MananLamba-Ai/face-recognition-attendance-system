import cv2
import csv
from datetime import datetime

# Attendance file
attendance_file = "attendance.csv"

# Load Haar model and trainer
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read("trainer.yml")

def mark_attendance(user_id):
    now = datetime.now()
    dt_string = now.strftime("%Y-%m-%d %H:%M:%S")

    with open(attendance_file, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([user_id, dt_string])

    print(f"Attendance marked for User: {user_id}")

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        face = gray[y:y+h, x:x+w]
        id, confidence = recognizer.predict(face)

        if confidence < 70:
            cv2.putText(frame, f"User {id}", (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
            mark_attendance(id)
        else:
            cv2.putText(frame, "Unknown", (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)

        cv2.rectangle(frame, (x, y), (x+w, y+h), (255,0,0), 2)

    cv2.imshow("Face Recognition Attendance", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
