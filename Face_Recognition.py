import cv2
import dlib
import face_recognition
import time

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("models/shape_predictor_68_face_landmarks.dat")

data = {
    "images": ["Data/Sample.png"],
    "ids": ["name_id"]
}

known_face_encodings = []
known_face_ids = []

for image_path, face_id in zip(data["images"], data["ids"]):
    image = face_recognition.load_image_file(image_path)
    face_encoding = face_recognition.face_encodings(image, num_jitters=10, model='large')[0]
    known_face_encodings.append(face_encoding)
    known_face_ids.append(face_id)

video_capture = cv2.VideoCapture(0)

pTime = time.time()

while True:
    ret, frame = video_capture.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = detector(gray)

    for face in faces:
        landmarks = predictor(gray, face)
        aligned_face = dlib.get_face_chip(frame, landmarks)
        face_encodings = face_recognition.face_encodings(aligned_face)

        if len(face_encodings) == 0:
            continue

        face_encoding = face_encodings[0]

        matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
        face_id = "Unknown"
        face_confidence = 0.0

        face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
        best_match_index = face_distances.argmin()

        if matches[best_match_index]:
            face_id = known_face_ids[best_match_index]
            scaling_factor = 1.3
            face_confidence = scaling_factor / (1.0 + scaling_factor * face_distances[best_match_index])

        if face_confidence < 0.5:
            cv2.rectangle(frame, (face.left(), face.top()), (face.right(), face.bottom()), (0, 255, 0), 2)
            cv2.putText(frame, "Maske", (face.left(), face.top() - 10),
                        cv2.FONT_HERSHEY_SCRIPT_SIMPLEX, 0.9, (0, 255, 0), 2)
            continue

        t = 5
        l = 30
        rt = 1

        cv2.rectangle(frame, (face.left(), face.top()), (face.right(), face.bottom()), (0, 255, 0), rt)

        cv2.rectangle(frame, (face.left(), face.top()), (face.right(), face.bottom()), (0, 255, 0), rt)

        cv2.line(frame, (face.left(), face.top()), (face.left() + l, face.top()), (0, 255, 0), t)
        cv2.line(frame, (face.left(), face.top()), (face.left(), face.top() + l), (0, 255, 0), t)

        cv2.line(frame, (face.right(), face.top()), (face.right() - l, face.top()), (0, 255, 0), t)
        cv2.line(frame, (face.right(), face.top()), (face.right(), face.top() + l), (0, 255, 0), t)

        cv2.line(frame, (face.left(), face.bottom()), (face.left() + l, face.bottom()), (0, 255, 0), t)
        cv2.line(frame, (face.left(), face.bottom()), (face.left(), face.bottom() - l), (0, 255, 0), t)

        cv2.line(frame, (face.right(), face.bottom()), (face.right() - l, face.bottom()), (0, 255, 0), t)
        cv2.line(frame, (face.right(), face.bottom()), (face.right(), face.bottom() - l), (0, 255, 0), t)

        cv2.rectangle(frame, (face.left(), face.top()), (face.right(), face.bottom()), (0, 255, 0), 2)
        cv2.putText(frame, f"{face_id} ({face_confidence:.2f})", (face.left(), face.top() - 10),
                    cv2.FONT_HERSHEY_COMPLEX, 0.9, (0, 255, 0), 2)

    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime
    cv2.putText(frame, f'FPS: {int(fps)}', (20, 70), cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 0), 2)

    cv2.imshow('Video', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video_capture.release()
cv2.destroyAllWindows()
