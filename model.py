from ultralytics import YOLO

model = YOLO("/Users/jomaricelis/code/machine learning/runs/classify/train12/weights/best.pt")  # Load your trained model


import streamlit as st
import cv2
import tempfile
import os
import numpy as np
from ultralytics import YOLO
import mediapipe as mp
from moviepy import *

import asyncio

try:
    asyncio.get_running_loop()
except RuntimeError:
    asyncio.set_event_loop(asyncio.new_event_loop())


# Initialize MediaPipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

# Load YOLOv11 Model
model = YOLO("/Users/jomaricelis/code/machine learning/runs/classify/train11/weights/best.pt")  # Update path

st.title("Attentiveness Detector")

# Upload video
uploaded_file = st.file_uploader("Upload a video", type=["mp4", "avi", "mov"])

if uploaded_file is not None:
    # Save uploaded file temporarily
    temp_dir = tempfile.TemporaryDirectory()
    input_video_path = os.path.join(temp_dir.name, uploaded_file.name)

    with open(input_video_path, "wb") as f:
        f.write(uploaded_file.read())

    st.video(input_video_path)

    # Process Video
    def process_video(video_path):
        cap = cv2.VideoCapture(video_path)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        output_path = os.path.join(temp_dir.name, "output.mp4")

        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        with mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=4, min_detection_confidence=0.2, min_tracking_confidence=0.5) as face_mesh:
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                # Convert frame to RGB for MediaPipe
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = face_mesh.process(rgb_frame)

                if results.multi_face_landmarks:
                    for face_landmarks in results.multi_face_landmarks:
                        x_min, y_min = float('inf'), float('inf')
                        x_max, y_max = float('-inf'), float('-inf')
                        h, w, _ = frame.shape

                        for lm in face_landmarks.landmark:
                            x, y = int(lm.x * w), int(lm.y * h)
                            x_min, y_min = min(x_min, x), min(y_min, y)
                            x_max, y_max = max(x_max, x), max(y_max, y)

                        # Expand bounding box
                        padding = 20
                        x_min, y_min = max(0, x_min - padding), max(0, y_min - padding)
                        x_max, y_max = min(w, x_max + padding), min(h, y_max + padding)

                        # Crop face region
                        face_roi = frame[y_min:y_max, x_min:x_max].copy()

                        if face_roi.size > 0:  # Ensure valid face region
                            yolo_results = model(face_roi)

                            if yolo_results:
                                class_name = yolo_results[0].names[yolo_results[0].probs.top1]
                                confidence = yolo_results[0].probs.top1conf

                                # Assign color based on classification
                                color = (0, 255, 0) if class_name == "Attentive" else (0, 0, 255)
                                cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), color, 2)
                                text = f"{class_name} ({confidence:.2f})"
                                cv2.putText(frame, text, (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

                        # Draw facial landmarks
                        mp_drawing.draw_landmarks(
                            frame, face_landmarks, mp_face_mesh.FACEMESH_TESSELATION,
                            landmark_drawing_spec=None,
                            connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_tesselation_style()
                        )

                out.write(frame)

        cap.release()
        out.release()
        return output_path


    if st.button("Process Video"):
        output_video_path = process_video(input_video_path)
        st.success("Processing Complete!")

        # Display processed video
        st.video(output_video_path)

        # Provide a download link
        with open(output_video_path, "rb") as f:
            st.download_button("Download Processed Video", f, file_name="processed.mp4")


