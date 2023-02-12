import pandas as pd
import numpy as np
import pickle
import mediapipe as mp
import cv2
import requests
import warnings
import sys

warnings.simplefilter("ignore", UserWarning)

if len(sys.argv) < 2:
    print("Please provide the key receiver's URL as an argument.")
    sys.exit()

server_url = sys.argv[1]

# Load detection model data
with open("hand_posture_3.pkl", "rb") as f:
    model = pickle.load(f)


def signal_keypress(key: str):
    requests.post(server_url, key.encode("UTF-8"))


cooldown = 0


def decide_direction(event_class, prob):
    global cooldown
    print(f"Class: {event_class}, Prob: {max(prob)}")
    cooldown -= 1
    if max(prob) > 0.6 and cooldown < 1:
        if event_class == "left_swipe":
            signal_keypress("left")
            cooldown = 10
        elif event_class == "right_swipe":
            signal_keypress("right")
            cooldown = 10


cap = cv2.VideoCapture(0)
mp_drawing = mp.solutions.drawing_utils
mp_holistic = mp.solutions.holistic

with mp_holistic.Holistic(
    min_detection_confidence=0.5, min_tracking_confidence=0.5
) as holistic:
    while cap.isOpened():
        ret, frame = cap.read()
        # Recolor Feed
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False
        # Make Detections
        results = holistic.process(image)
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        # Export coordinates
        try:
            # Extract Pose landmarks
            left_hand = results.left_hand_landmarks.landmark
            left_hand_row = list(
                np.array(
                    [
                        [landmark.x, landmark.y, landmark.z, landmark.visibility]
                        for landmark in left_hand
                    ]
                ).flatten()
            )
            # Extract Face landmarks
            right_hand = results.right_hand_landmarks.landmark
            right_hand_row = list(
                np.array(
                    [
                        [landmark.x, landmark.y, landmark.z, landmark.visibility]
                        for landmark in right_hand
                    ]
                ).flatten()
            )
            # Pose
            pose = results.pose_landmarks.landmark
            pose_row = list(
                np.array(
                    [
                        [landmark.x, landmark.y, landmark.z, landmark.visibility]
                        for landmark in pose
                    ]
                ).flatten()
            )
            # Concate rows
            row = left_hand_row + right_hand_row + pose_row
            # Make Detections
            X = pd.DataFrame([row])
            body_language_class = model.predict(X)[0]
            body_language_prob = model.predict_proba(X)[0]
            decide_direction(body_language_class, body_language_prob)
            # Grab ear coords
            coords = tuple(
                np.multiply(
                    np.array(
                        (
                            results.pose_landmarks.landmark[
                                mp_holistic.PoseLandmark.LEFT_EAR
                            ].x,
                            results.pose_landmarks.landmark[
                                mp_holistic.PoseLandmark.LEFT_EAR
                            ].y,
                        )
                    ),
                    [640, 480],
                ).astype(int)
            )
            # Get status box
            cv2.rectangle(image, (0, 0), (250, 60), (245, 117, 16), -1)
            # Display Class
            cv2.putText(
                image,
                "CLASS",
                (95, 12),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 0, 0),
                1,
                cv2.LINE_AA,
            )
            cv2.putText(
                image,
                body_language_class.split(" ")[0],
                (90, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )
            # Display Probability
            cv2.putText(
                image,
                "PROB",
                (15, 12),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 0, 0),
                1,
                cv2.LINE_AA,
            )
            cv2.putText(
                image,
                str(round(body_language_prob[np.argmax(body_language_prob)], 2)),
                (10, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )
        except:
            pass
        cv2.imshow("Puppeteer", image)
        if cv2.waitKey(10) & 0xFF == ord("q"):
            break
