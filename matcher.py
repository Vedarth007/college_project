import time
from PIL import Image, ImageDraw
import numpy as np
import numpy as np
import mediapipe as mp
from streamlit_webrtc import VideoTransformerBase
from pose_utils import calculate_angle

mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

class PoseMatcher(VideoTransformerBase):
    def __init__(self, reference_angle):
        self.pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)
        self.start_time = time.time()
        self.delay = 7
        self.match_found = False
        self.feedback = "⏳ Warming up..."
        self.last_frame = None
        self.match_attempt_time = None
        self.reference_angle = reference_angle

    def transform(self, frame):
        # If already matched, return the frozen frame
        if self.match_found and self.last_frame is not None:
            return self.last_frame

        img = frame.to_ndarray(format="bgr24")
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        elapsed = time.time() - self.start_time

        # Warm-up countdown
        if elapsed < self.delay:
            countdown = int(self.delay - elapsed)
            cv2.putText(img, f"Get ready... {countdown}s", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
            self.feedback = "⏳ Warming up..."

        else:
            # Pose detection
            res = self.pose.process(img_rgb)
            if res.pose_landmarks:
                mp_drawing.draw_landmarks(img, res.pose_landmarks, mp_pose.POSE_CONNECTIONS)
                lm = res.pose_landmarks.landmark

                # Extract points
                sh = [lm[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,
                      lm[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
                el = [lm[mp_pose.PoseLandmark.LEFT_ELBOW.value].x,
                      lm[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
                wr = [lm[mp_pose.PoseLandmark.LEFT_WRIST.value].x,
                      lm[mp_pose.PoseLandmark.LEFT_WRIST.value].y]

                # Calculate angle and difference
                user_ang = calculate_angle(sh, el, wr)
                diff = abs(user_ang - self.reference_angle)

                # Check for match held for 3 seconds
                if diff < 5:
                    if not self.match_attempt_time:
                        self.match_attempt_time = time.time()
                    elif time.time() - self.match_attempt_time > 3:
                        self.feedback = f"✅ Pose matched! Angle: {int(user_ang)}°"
                        self.match_found = True
                else:
                    self.match_attempt_time = None
                    self.feedback = f"❌ Try again. Angle: {int(user_ang)}°"

                # Overlay feedback
                cv2.putText(img, self.feedback, (10, 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

        self.last_frame = img.copy()
        return img
