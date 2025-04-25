import cv2
import mediapipe as mp
import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase
import av
import numpy as np
from pose_utils import calculate_angle

st.set_page_config(layout="wide")
st.title("🧘 Pose Matcher")

# Load and process reference image
ref_path = "ref_pose.jpg"
ref_img = cv2.imread(ref_path)
if ref_img is None:
    st.error("Reference image not found.")
    st.stop()

ref_img_rgb = cv2.cvtColor(ref_img, cv2.COLOR_BGR2RGB)

mp_pose = mp.solutions.pose
with mp_pose.Pose(static_image_mode=True) as pose:
    res = pose.process(ref_img_rgb)
    if not res.pose_landmarks:
        st.error("No pose detected in reference image.")
        st.stop()

    lm = res.pose_landmarks.landmark
    shoulder = [lm[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,
                lm[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
    elbow = [lm[mp_pose.PoseLandmark.LEFT_ELBOW.value].x,
             lm[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
    wrist = [lm[mp_pose.PoseLandmark.LEFT_WRIST.value].x,
             lm[mp_pose.PoseLandmark.LEFT_WRIST.value].y]

    reference_angle = calculate_angle(shoulder, elbow, wrist)

# PoseMatcher with freeze-on-match logic
class PoseMatcher(VideoTransformerBase):
    def __init__(self):
        self.pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)
        self.freeze = False
        self.frozen_frame = None

    def transform(self, frame: av.VideoFrame) -> np.ndarray:
        if self.freeze and self.frozen_frame is not None:
            return self.frozen_frame

        img = frame.to_ndarray(format="bgr24")
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = self.pose.process(img_rgb)

        if results.pose_landmarks:
            mp_drawing = mp.solutions.drawing_utils
            mp_drawing.draw_landmarks(img, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

            lm = results.pose_landmarks.landmark
            shoulder = [lm[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,
                        lm[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
            elbow = [lm[mp_pose.PoseLandmark.LEFT_ELBOW.value].x,
                     lm[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
            wrist = [lm[mp_pose.PoseLandmark.LEFT_WRIST.value].x,
                     lm[mp_pose.PoseLandmark.LEFT_WRIST.value].y]

            current_angle = calculate_angle(shoulder, elbow, wrist)

            cv2.putText(img, f'Angle: {int(current_angle)} deg',
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            # Check angle match
            if abs(current_angle - reference_angle) < 15:
                match_status = "✅ Match!"
                color = (0, 255, 0)
                self.freeze = True
                self.frozen_frame = img.copy()
            else:
                match_status = "❌ Not Matching"
                color = (0, 0, 255)

            cv2.putText(img, match_status, (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

        return img

# UI layout
col1, col2 = st.columns(2)

with col1:
    st.subheader("📸 Reference Pose")
    st.image(ref_img_rgb, caption="Reference Pose", use_column_width=True)
    st.success(f"Reference Elbow Angle: {int(reference_angle)}°")

with col2:
    st.subheader("🎥 Live Camera")
    webrtc_streamer(
        key="pose_matcher",
        video_transformer_factory=PoseMatcher,
        media_stream_constraints={"video": True, "audio": False},
        async_processing=True
    )
