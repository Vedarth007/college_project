import numpy as np
import mediapipe as mp
from PIL import Image
import streamlit as st
from streamlit_webrtc import webrtc_streamer

from pose_utils import calculate_angle
from matcher import PoseMatcher

st.set_page_config(layout="wide")
st.title("🧘 Pose Matcher")

uploaded_file = st.file_uploader("Upload Reference Pose Image", type=["jpg", "png"])
if not uploaded_file:
    st.warning("Please upload a reference image to start.")
    st.stop()

# Use PIL to read and display the uploaded image
ref_img = Image.open(uploaded_file).convert("RGB")
ref_img_np = np.array(ref_img)
st.image(ref_img, caption="Your reference pose", use_column_width=True)

# Extract pose from image using MediaPipe
with mp.solutions.pose.Pose(static_image_mode=True) as pose_ref:
    results = pose_ref.process(ref_img_np)
    if not results.pose_landmarks:
        st.error("No pose detected in uploaded image.")
        st.stop()
    lm = results.pose_landmarks.landmark
    def pt(i): return [lm[i].x, lm[i].y]
    shoulder, elbow, wrist = pt(mp.solutions.pose.PoseLandmark.LEFT_SHOULDER.value), \
                             pt(mp.solutions.pose.PoseLandmark.LEFT_ELBOW.value), \
                             pt(mp.solutions.pose.PoseLandmark.LEFT_WRIST.value)
    reference_angle = calculate_angle(shoulder, elbow, wrist)
    st.success(f"Reference Elbow Angle: {int(reference_angle)}°")

# Live camera tab
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
