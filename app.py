import cv2
import mediapipe as mp
import streamlit as st
from streamlit_webrtc import webrtc_streamer
from pose_utils import calculate_angle
from matcher import PoseMatcher

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
        video_transformer_factory=lambda: PoseMatcher(reference_angle)
    )
