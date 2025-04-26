import cv2
import numpy as np
import mediapipe as mp
from math import degrees
import streamlit as st
from streamlit_webrtc import webrtc_streamer

from pose_utils import calculate_angle
from matcher import PoseMatcher

st.set_page_config(layout="wide")
st.title("🧘 Pose Matcher")

# Column 1: Upload and process reference image
uploaded_file = st.file_uploader("Upload Reference Pose Image", type=["jpg", "png"])
if not uploaded_file:
    st.warning("Please upload a reference image to start.")
    st.stop()

# Decode to OpenCV format
file_bytes = np.frombuffer(uploaded_file.read(), np.uint8)
ref_img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
ref_img_rgb = cv2.cvtColor(ref_img, cv2.COLOR_BGR2RGB)
st.image(ref_img_rgb, caption="Your reference pose", use_column_width=True)

# Extract reference elbow angle
with mp.solutions.pose.Pose(static_image_mode=True) as pose_ref:
    res = pose_ref.process(ref_img_rgb)
    if not res.pose_landmarks:
        st.error("No pose detected in uploaded image.")
        st.stop()
    lm = res.pose_landmarks.landmark
    def pt(i): return [lm[i].x, lm[i].y]
    shoulder, elbow, wrist = pt(mp_pose.PoseLandmark.LEFT_SHOULDER.value), \
                             pt(mp_pose.PoseLandmark.LEFT_ELBOW.value), \
                             pt(mp_pose.PoseLandmark.LEFT_WRIST.value)
    reference_angle = calculate_angle(shoulder, elbow, wrist)
    st.success(f"Reference Elbow Angle: {int(reference_angle)}°")

# Column 2: Live camera + matching logic
tab1, tab2 = st.tabs(["Reference", "Live Camera"])
with tab2:
    st.subheader("📷 Your Camera")
    ctx = webrtc_streamer(
        key="pose_matcher",
        video_transformer_factory=lambda: PoseMatcher(reference_angle),
        media_stream_constraints={"video": True, "audio": False},
        async_processing=True
    )
    feedback_placeholder = st.empty()
    if ctx.video_transformer:
        if ctx.video_transformer.match_found:
            feedback_placeholder.success(ctx.video_transformer.feedback)
        else:
            feedback_placeholder.info(ctx.video_transformer.feedback)
