import cv2
import numpy as np
from scipy.spatial import distance as dist

def detect_eyes_blink(landmarks):
    # Calculate the eye aspect ratio
    def eye_aspect_ratio(eye):
        A = dist.euclidean(eye[1], eye[5])
        B = dist.euclidean(eye[2], eye[4])
        C = dist.euclidean(eye[0], eye[3])
        ear = (A + B) / (2.0 * C)
        return ear

    # Get the landmarks for the left and right eyes
    left_eye = landmarks[36:42]
    right_eye = landmarks[42:48]

    # Calculate the eye aspect ratio for both eyes
    left_ear = eye_aspect_ratio(left_eye)
    right_ear = eye_aspect_ratio(right_eye)

    # Average the eye aspect ratio
    ear = (left_ear + right_ear) / 2.0

    # Check if the eye aspect ratio is below the blink threshold
    if ear < 0.3:  # You might need to adjust this threshold
        return True
    return False

def detect_liveness(frame):
    # Load face detector and landmark predictor
    face_detector = cv2.dnn.readNetFromCaffe("deploy.prototxt", "res10_300x300_ssd_iter_140000.caffemodel")
    landmark_predictor = cv2.face.createFacemarkLBF()
    landmark_predictor.loadModel("lbfmodel.yaml")

    # Preprocess the frame
    blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0))

    # Detect faces
    face_detector.setInput(blob)
    detections = face_detector.forward()

    for i in range(0, detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > 0.5:
            box = detections[0, 0, i, 3:7] * np.array([frame.shape[1], frame.shape[0], frame.shape[1], frame.shape[0]])
            (startX, startY, endX, endY) = box.astype("int")

            # Extract the face ROI
            face = frame[startY:endY, startX:endX]

            # Detect landmarks
            _, landmarks = landmark_predictor.fit(frame, np.array([startX, startY, endX-startX, endY-startY]))

            if len(landmarks) > 0:
                landmarks = landmarks[0][0]

                # Check for eye blink
                if detect_eyes_blink(landmarks):
                    return True  # Liveness detected

    return False  # No liveness detected

# Main loop
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    liveness = detect_liveness(frame)

    # Display result
    cv2.putText(frame, f"Liveness: {'Detected' if liveness else 'Not Detected'}", 
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    cv2.imshow("Liveness Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()