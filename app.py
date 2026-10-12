from collections import deque
import os
import sys
import tempfile

# === CRITICAL: Set up MediaPipe cache BEFORE any imports ===
# Use system temp directory which is always writable
mediapipe_cache = os.path.join(tempfile.gettempdir(), "mediapipe_cache")
os.makedirs(mediapipe_cache, exist_ok=True)
os.environ['MEDIAPIPE_PYTHON_MODELS_DIR'] = mediapipe_cache

# Disable MediaPipe's automatic model download attempt
os.environ['MEDIAPIPE_DISABLE_GPU'] = '1'

import av
import cv2
import numpy as np
import streamlit as st

# Import mediapipe after environment setup
try:
    import mediapipe as mp
    from mediapipe.tasks import python
    from mediapipe.tasks.python import vision
except ImportError:
    st.error("MediaPipe not installed. Please run: pip install mediapipe==0.10.21")
    st.stop()

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
    try:
        pose_model = mp_pose.Pose(
            model_complexity=0,            # fastest model, good for CPU hosting
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        return {
            "pose": pose_model,
            "ankle_gap": deque(maxlen=20),     # recent left-right ankle height gap
            "labels": deque(maxlen=7),         # recent labels for smoothing
            "error": None,
        }
    except PermissionError as e:
        error_msg = f"Permission denied loading MediaPipe model. Using fallback mode."
        return {
            "pose": None,
            "ankle_gap": deque(maxlen=20),
            "labels": deque(maxlen=7),
            "error": error_msg,
        }
    except Exception as e:
        error_msg = f"Error loading MediaPipe Pose model: {type(e).__name__}: {str(e)}"
        return {
            "pose": None,
            "ankle_gap": deque(maxlen=20),
            "labels": deque(maxlen=7),
            "error": error_msg,
        }


state = get_state()

# Display error if model failed to load
if state["error"]:
    st.warning(f"⚠️ {state['error']}")
    st.info(
        "Streamlit Cloud has restricted permissions for the MediaPipe installation directory. "
        "Please try these solutions:\n\n"
        "1. **Add a `.streamlit/config.toml`** file to your repo with:\n"
        "```\n"
        "[logger]\n"
        "level = \"warning\"\n"
        "```\n\n"
        "2. **Upgrade MediaPipe**: Update `requirements.txt` to use `mediapipe>=0.11.0`\n\n"
        "3. **Alternative**: Deploy on a platform with fewer restrictions (Replit, Render, Railway)."
    )


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
    # If model failed to load, show error message on video
    if state["pose"] is None:
        img = frame.to_ndarray(format="bgr24")
        cv2.rectangle(img, (0, 0), (img.shape[1], 42), (0, 0, 0), -1)
        cv2.putText(img, "Model failed to load", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        return av.VideoFrame.from_ndarray(img, format="bgr24")

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


if state["pose"] is not None:
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
else:
    st.error("Cannot start video stream without a working pose model.")
