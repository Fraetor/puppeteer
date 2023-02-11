import tkinter as tk
import customtkinter as ck
import pandas as pd
import numpy as np
import pickle
import mediapipe as mp
import cv2
from PIL import Image, ImageTk
from landmarks import landmarks
import requests

server_url = "http://127.0.0.1:8888"
confidence = 0.7

# Window setup
window = tk.Tk()
window.geometry("500x700")
window.title("Puppeteer")
ck.set_appearance_mode("light")

# Metric labels
action_label = ck.CTkLabel(window, height=40, width=120, text_color="black", padx=10)
action_label.place(x=10, y=1)
action_label.configure(text="Action")
counter_label = ck.CTkLabel(window, height=40, width=120, text_color="black", padx=10)
counter_label.place(x=195, y=1)
counter_label.configure(text="Count")
prob_label = ck.CTkLabel(window, height=40, width=120, text_color="black", padx=10)
prob_label.place(x=370, y=1)
prob_label.configure(text="Prob.")
class_box = ck.CTkLabel(
    window, height=40, width=120, text_color="white", fg_color="blue"
)
class_box.place(x=10, y=41)
class_box.configure(text="0")
counter_box = ck.CTkLabel(
    window, height=40, width=120, text_color="white", fg_color="blue"
)
counter_box.place(x=195, y=41)
counter_box.configure(text="0")
prob_box = ck.CTkLabel(
    window, height=40, width=120, text_color="white", fg_color="blue"
)
prob_box.place(x=370, y=41)
prob_box.configure(text="0")


# Reset button
def reset_counter():
    global counter
    counter = 0


button = ck.CTkButton(
    window,
    text="Reset",
    command=reset_counter,
    height=40,
    width=100,
    text_color="white",
    fg_color="blue",
)
button.place(x=200, y=600)

# Video display frame
frame = tk.Frame(height=480, width=480)
frame.place(x=10, y=90)
lmain = tk.Label(frame)
lmain.place(x=0, y=0)
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(min_tracking_confidence=0.5, min_detection_confidence=0.5)

# Load detection model data
with open("body_language.pkl", "rb") as f:
    model = pickle.load(f)

cap = cv2.VideoCapture(0)
current_stage = "front"
counter = 0
bodylang_prob = np.array([0, 0])
bodylang_class = ""


def signal_keypress(key: str):
    requests.post(server_url, key.encode("UTF-8"))


def detect():
    global current_stage
    global counter
    global bodylang_class
    global bodylang_prob
    ret, frame = cap.read()
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(image)
    mp_drawing.draw_landmarks(
        image,
        results.pose_landmarks,
        mp_pose.POSE_CONNECTIONS,
        mp_drawing.DrawingSpec(color=(106, 13, 173), thickness=4, circle_radius=5),
        mp_drawing.DrawingSpec(color=(255, 102, 0), thickness=5, circle_radius=10),
    )
    try:
        row = (
            np.array(
                [
                    [res.x, res.y, res.z, res.visibility]
                    for res in results.pose_landmarks.landmark
                ]
            )
            .flatten()
            .tolist()
        )
        X = pd.DataFrame([row], columns=landmarks)
        bodylang_prob = model.predict_proba(X)[0]
        bodylang_class = model.predict(X)[0]
        if (
            bodylang_class == "left_head"
            and bodylang_prob[bodylang_prob.argmax()] > confidence
        ):
            if current_stage != "left":
                current_stage = "left"
                signal_keypress("left")
                counter += 1
        elif (
            bodylang_class == "right_head"
            and bodylang_prob[bodylang_prob.argmax()] > confidence
        ):
            if current_stage != "right":
                current_stage = "right"
                signal_keypress("right")
                counter += 1
        else:
            current_stage = "front"
    except Exception as e:
        print(e)
    img = image[:, :480, :]
    imgarr = Image.fromarray(img)
    imgtk = ImageTk.PhotoImage(imgarr)
    lmain.imgtk = imgtk
    lmain.configure(image=imgtk)
    lmain.after(10, detect)
    counter_box.configure(text=counter)
    prob_box.configure(text=bodylang_prob[bodylang_prob.argmax()])
    class_box.configure(text=current_stage)


detect()
window.mainloop()
