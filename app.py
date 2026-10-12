from collections import deque
import os

# Set MediaPipe model cache to a writable directory before importing
os.environ['MEDIAPIPE_PYTHON_MODELS_DIR'] = os.path.expanduser('~/.mediapipe_models')

import av
import cv2
import numpy as np
import streamlit as st
import mediapipe as mp
from streamlit_webrtc import webrtc_streamer

st.set_page_config(page_title="Live Human Activity Recognition", page_icon="🏃")
st.title("🏃 Live Human Activity Recognition")
st.write(
    "Allow camera access and keep your **full body** in view. "
    "The app detects body landmarks (MediaPipe Pose) and labels your activity live: "
    "Standing, Sitting, Walking or Laying."
)

mp_pose = mp.solutions.pose
mp_draw = mp.solutions.drawing_utils
L = mp_pose.PoseLandmark


@st.cache_resource
def get_state():
    """One shared pose model + small history buffers (kept across reruns)."""
    return {
        "pose": mp_pose.Pose(
            model_complexity=0,            # fastest model, good for CPU hosting
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        ),
        "ankle_gap": deque(maxlen=20),     # recent left-right ankle height gap
        "labels": deque(maxlen=7),         # recent labels for smoothing
    }


state = get_state()


def classify(lm, st_):
    """Rule-based activity from pose landmarks (image coords: y grows downward)."""
    def pt(i):
        return np.array([lm[i].x, lm[i].y])

    key_points = [L.LEFT_SHOULDER, L.RIGHT_SHOULDER, L.LEFT_HIP, L.RIGHT_HIP,
                  L.LEFT_KNEE, L.RIGHT_KNEE]
    if min(lm[i].visibility for i in key_points) < 0.5:
        return "Show full body"

    shoulder = (pt(L.LEFT_SHOULDER) + pt(L.RIGHT_SHOULDER)) / 2
    hip = (pt(L.LEFT_HIP) + pt(L.RIGHT_HIP)) / 2
    knee = (pt(L.LEFT_KNEE) + pt(L.RIGHT_KNEE)) / 2
    torso = hip - shoulder
    torso_len = np.linalg.norm(torso) + 1e-6

    # 1) Laying: torso is closer to horizontal than vertical
    if abs(torso[0]) > abs(torso[1]):
        st_["ankle_gap"].clear()
        return "Laying"

    # 2) Walking: ankles keep swapping height (large variation over recent frames)
    gap = (lm[L.LEFT_ANKLE].y - lm[L.RIGHT_ANKLE].y) / torso_len
    st_["ankle_gap"].append(gap)
    if len(st_["ankle_gap"]) >= 15 and np.std(st_["ankle_gap"]) > 0.08:
        return "Walking"

    # 3) Sitting vs Standing: how far the knees hang below the hips
    thigh_drop = (knee[1] - hip[1]) / torso_len
    return "Sitting" if thigh_drop < 0.6 else "Standing"


def video_frame_callback(frame):
    img = frame.to_ndarray(format="bgr24")
    result = state["pose"].process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

    label = "No person detected"
    if result.pose_landmarks:
        mp_draw.draw_landmarks(img, result.pose_landmarks, mp_pose.POSE_CONNECTIONS)
        label = classify(result.pose_landmarks.landmark, state)

    state["labels"].append(label)
    shown = max(set(state["labels"]), key=list(state["labels"]).count)

    cv2.rectangle(img, (0, 0), (img.shape[1], 42), (0, 0, 0), -1)
    cv2.putText(img, f"Activity: {shown}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
    return av.VideoFrame.from_ndarray(img, format="bgr24")


webrtc_streamer(
    key="live-har",
    video_frame_callback=video_frame_callback,
    media_stream_constraints={"video": {"width": 640, "height": 480}, "audio": False},
    rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
    async_processing=True,
)

st.caption(
    "Tips: use good lighting, stand 2–3 m from the camera, keep head to feet visible. "
    "Walk sideways across the frame for best walking detection."
)
