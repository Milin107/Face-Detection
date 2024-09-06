import cv2
import mediapipe as mp
import numpy as np

# Initialize Mediapipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1, min_detection_confidence=0.5)

# Initialize OpenCV VideoCapture for accessing the camera
cap = cv2.VideoCapture(0)

# Define the indices for specific landmarks
# Mediapipe landmarks for facial features (for mouth and head tilt detection)
UPPER_LIP_INDEX = 13
LOWER_LIP_INDEX = 14
LEFT_MOUTH_CORNER = 61
RIGHT_MOUTH_CORNER = 291
FOREHEAD_INDEX = 10
CHIN_INDEX = 152

# Yawn detection threshold (based on mouth height/width ratio)
YAWN_THRESHOLD = 0.6
# Head down detection threshold (based on pixel height in the frame)
HEAD_DOWN_THRESHOLD = 320

# Start video processing loop
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("Failed to capture image")
        break

    # Flip the frame horizontally for a natural (mirror-like) view
    frame = cv2.flip(frame, 1)

    # Convert the frame color to RGB (from BGR as OpenCV reads)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process the frame to detect facial landmarks
    results = face_mesh.process(rgb_frame)

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            h, w, _ = frame.shape
            landmarks = np.array([(int(point.x * w), int(point.y * h)) for point in face_landmarks.landmark])

            # Head down detection based on the position of forehead and chin
            forehead_y = landmarks[FOREHEAD_INDEX][1]
            chin_y = landmarks[CHIN_INDEX][1]
            head_center_y = (forehead_y + chin_y) / 2
            if head_center_y > HEAD_DOWN_THRESHOLD:
                cv2.putText(frame, "Head Down Detected!", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

            # Yawn detection based on the ratio of mouth height to width
            upper_lip = landmarks[UPPER_LIP_INDEX]
            lower_lip = landmarks[LOWER_LIP_INDEX]
            mouth_height = np.linalg.norm(upper_lip - lower_lip)
            mouth_width = np.linalg.norm(landmarks[LEFT_MOUTH_CORNER] - landmarks[RIGHT_MOUTH_CORNER])

            if mouth_height / mouth_width > YAWN_THRESHOLD:
                cv2.putText(frame, "Yawn Detected!", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

            # Draw facial landmarks (optional visualization)
            for idx, point in enumerate(landmarks):
                if idx in [UPPER_LIP_INDEX, LOWER_LIP_INDEX, LEFT_MOUTH_CORNER, RIGHT_MOUTH_CORNER, FOREHEAD_INDEX, CHIN_INDEX]:
                    cv2.circle(frame, point, 2, (255, 0, 0), -1)

    else:
        # If no face is detected, show a warning
        cv2.putText(frame, "No face detected or camera blocked!", (50, 200), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    # Display the frame with detection
    cv2.imshow('Head, Yawn, and Camera Blockage Detection', frame)

    # Press 'q' to exit the loop
    if cv2.waitKey(5) & 0xFF == ord('q'):
        break

# Release the camera and close all windows
cap.release()
cv2.destroyAllWindows()
