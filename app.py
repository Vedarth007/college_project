import time
import cv2
import numpy as np
import mediapipe as mp
import streamlit as st
from streamlit_webrtc import VideoTransformerBase, webrtc_streamer
from matcher import PoseMatcher  # Import PoseMatcher from matcher.py

# --- Setup MediaPipe ---
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

st.set_page_config(layout="wide")
st.title("🧘 Pose Matcher")

# --- Column 1: Upload and process reference image ---
uploaded_file = st.file_uploader("Upload Reference Pose Image", type=["jpg", "png"])
if not uploaded_file:
    st.warning("Please upload a reference image to start.")
    st.stop()

# decode to OpenCV and show
file_bytes = np.frombuffer(uploaded_file.read(), np.uint8)
ref_img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
ref_img_rgb = cv2.cvtColor(ref_img, cv2.COLOR_BGR2RGB)
st.image(ref_img_rgb, caption="Your reference pose", use_column_width=True)

# extract elbow angle from reference
with mp_pose.Pose(static_image_mode=True) as pose_ref:
    res = pose_ref.process(ref_img_rgb)
    if not res.pose_landmarks:
        st.error("No pose detected in uploaded image.")
        st.stop()
    lm = res.pose_landmarks.landmark
    def pt(i): return [lm[i].x, lm[i].y]
    shoulder, elbow, wrist = pt(mp_pose.PoseLandmark.LEFT_SHOULDER.value), \
                             pt(mp_pose.PoseLandmark.LEFT_ELBOW.value), \
                             pt(mp_pose.PoseLandmark.LEFT_WRIST.value)
    def angle(a, b, c):
        a, b, c = np.array(a), np.array(b), np.array(c)
        rad = np.arccos(np.clip(np.dot(a - b, c - b) / 
                                 (np.linalg.norm(a - b) * np.linalg.norm(c - b)), -1.0, 1.0))
        return np.degrees(rad)
    reference_angle = angle(shoulder, elbow, wrist)
    st.success(f"Reference Elbow Angle: {int(reference_angle)}°")

# --- Column 2: Live camera + matching logic ---
class PoseMatcherApp(VideoTransformerBase):
    def __init__(self, reference_angle):
        self.pose = mp_pose.Pose()
        self.start_time = time.time()
        self.delay = 7
        self.match_found = False
        self.feedback = "⏳ Warming up..."
        self.last_frame = None
        self.match_attempt_time = None
        self.reference_angle = reference_angle

    def angle(self, a, b, c):
        a, b, c = np.array(a), np.array(b), np.array(c)
        rad = np.arccos(np.clip(np.dot(a - b, c - b) /
                                (np.linalg.norm(a - b) * np.linalg.norm(c - b)), -1.0, 1.0))
        return np.degrees(rad)

    def transform(self, frame):
        img = frame.to_ndarray(format="bgr24")
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        elapsed = time.time() - self.start_time

        # freeze last frame once matched
        if self.match_found:
            return self.last_frame

        # warming-up countdown
        if elapsed < self.delay:
            countdown = int(self.delay - elapsed)
            cv2.putText(img, f"Get ready... {countdown}s", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
            self.feedback = "⏳ Warming up..."
        else:
            # pose detection & matching
            res = self.pose.process(img_rgb)
            if res.pose_landmarks:
                mp_drawing.draw_landmarks(img, res.pose_landmarks, mp_pose.POSE_CONNECTIONS)
                lm = res.pose_landmarks.landmark
                sh = [lm[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,
                      lm[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
                el = [lm[mp_pose.PoseLandmark.LEFT_ELBOW.value].x,
                      lm[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
                wr = [lm[mp_pose.PoseLandmark.LEFT_WRIST.value].x,
                      lm[mp_pose.PoseLandmark.LEFT_WRIST.value].y]

                user_ang = self.angle(sh, el, wr)
                diff = abs(user_ang - self.reference_angle)

                if diff < 5:
                    if not self.match_attempt_time:
                        self.match_attempt_time = time.time()
                    elif time.time() - self.match_attempt_time > 3:
                        self.feedback = f"✅ Pose matched! Angle: {int(user_ang)}°"
                        self.match_found = True
                else:
                    self.match_attempt_time = None
                    self.feedback = f"❌ Try again. Angle: {int(user_ang)}°"

                cv2.putText(img, self.feedback, (10, 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

        self.last_frame = img.copy()
        return img

# --- Render stream + conditional feedback below camera ---
st.subheader("📷 Your Camera")
ctx = webrtc_streamer(key="pose_matcher", video_transformer_factory=lambda: PoseMatcherApp(reference_angle))

# Placeholder for feedback
feedback_placeholder = st.empty()

# If pose is matched, show success message in new box below the camera feed
if ctx.video_transformer:
    if ctx.video_transformer.match_found:
        feedback_placeholder.success(ctx.video_transformer.feedback)  # Success message for matched pose
    else:
        feedback_placeholder.info(ctx.video_transformer.feedback)  # Keep showing "Warming up..." message

