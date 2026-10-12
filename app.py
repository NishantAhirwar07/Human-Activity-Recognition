from collections import deque
import os
import sys
import tempfile
import warnings

# Suppress MediaPipe warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppress TensorFlow logs

# === CRITICAL: Set up MediaPipe cache BEFORE any imports ===
# Use system temp directory which is always writable
mediapipe_cache = os.path.join(tempfile.gettempdir(), "mediapipe_models")
os.makedirs(mediapipe_cache, exist_ok=True)
os.environ['MEDIAPIPE_PYTHON_MODELS_DIR'] = mediapipe_cache
os.environ['MEDIAPIPE_DISABLE_GPU'] = '1'

import av
import cv2
import numpy as np
import streamlit as st

st.set_page_config(page_title="Live Human Activity Recognition", page_icon="🏃")
st.title("🏃 Live Human Activity Recognition")
st.write(
    "Allow camera access and keep your **full body** in view. "
    "The app detects body landmarks (MediaPipe Pose) and labels your activity live: "
    "Standing, Sitting, Walking or Laying."
)

# Import mediapipe after environment setup and warnings suppression
try:
    import mediapipe as mp
    from mediapipe import solutions
    from mediapipe.framework.formats import landmark_pb2
except ImportError as e:
    st.error(f"❌ MediaPipe import failed: {e}")
    st.info("Please reinstall dependencies: `pip install --upgrade mediapipe`")
    st.stop()

from streamlit_webrtc import webrtc_streamer

mp_pose = solutions.pose
mp_drawing = solutions.drawing_utils
L = mp_pose.PoseLandmark

pose_model = None
model_error = None


@st.cache_resource
def initialize_pose_model():
    """Initialize MediaPipe Pose model with error handling."""
    global pose_model, model_error
    
    try:
        # Try to create Pose with lite model (fastest, most compatible)
        model = mp_pose.Pose(
            model_complexity=0,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
            static_image_mode=False,
        )
        return model, None
    
    except PermissionError as e:
        error_msg = (
            "🔒 **Permission Denied**: MediaPipe cannot access model files on Streamlit Cloud. "
            "This is a platform limitation, not a code issue."
        )
        return None, error_msg
    
    except Exception as e:
        error_msg = f"❌ Error initializing MediaPipe Pose: {type(e).__name__}: {str(e)[:100]}"
        return None, error_msg


# Initialize model
pose_model, model_error = initialize_pose_model()

# Display error and solutions
if model_error:
    st.warning(model_error)
    with st.expander("ℹ️ Solutions", expanded=True):
        st.markdown("""
        **Option 1: Refresh the page** (sometimes the model cache loads on retry)
        
        **Option 2: Deploy on a less-restricted platform:**
        - [Render](https://render.com) - Free tier available
        - [Replit](https://replit.com) - Great for Python apps
        - [Railway](https://railway.app) - Simple deployment
        - [PythonAnywhere](https://www.pythonanywhere.com) - Python-focused
        
        **Option 3: Run locally**
        ```bash
        git clone <repo-url>
        cd Human-Activity-Recognition
        pip install -r requirements.txt
        streamlit run app.py
        ```
        """)


def classify(landmarks, history_state):
    """Rule-based activity classification from pose landmarks."""
    def pt(i):
        return np.array([landmarks[i].x, landmarks[i].y])

    key_points = [
        L.LEFT_SHOULDER, L.RIGHT_SHOULDER, 
        L.LEFT_HIP, L.RIGHT_HIP,
        L.LEFT_KNEE, L.RIGHT_KNEE
    ]
    
    # Check visibility
    if min(landmarks[i].visibility for i in key_points) < 0.5:
        return "Show full body"

    shoulder = (pt(L.LEFT_SHOULDER) + pt(L.RIGHT_SHOULDER)) / 2
    hip = (pt(L.LEFT_HIP) + pt(L.RIGHT_HIP)) / 2
    knee = (pt(L.LEFT_KNEE) + pt(L.RIGHT_KNEE)) / 2
    torso = hip - shoulder
    torso_len = np.linalg.norm(torso) + 1e-6

    # 1) Laying: torso is closer to horizontal than vertical
    if abs(torso[0]) > abs(torso[1]):
        history_state["ankle_gap"].clear()
        return "Laying"

    # 2) Walking: ankles keep swapping height
    gap = (landmarks[L.LEFT_ANKLE].y - landmarks[L.RIGHT_ANKLE].y) / torso_len
    history_state["ankle_gap"].append(gap)
    
    if len(history_state["ankle_gap"]) >= 15:
        gap_std = np.std(list(history_state["ankle_gap"]))
        if gap_std > 0.08:
            return "Walking"

    # 3) Sitting vs Standing
    thigh_drop = (knee[1] - hip[1]) / torso_len
    return "Sitting" if thigh_drop < 0.6 else "Standing"


def video_frame_callback(frame):
    """Process video frames for pose detection."""
    if pose_model is None:
        img = frame.to_ndarray(format="bgr24")
        cv2.rectangle(img, (0, 0), (img.shape[1], 42), (200, 50, 50), -1)
        cv2.putText(
            img, "Model unavailable", (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2
        )
        return av.VideoFrame.from_ndarray(img, format="bgr24")

    img = frame.to_ndarray(format="bgr24")
    rgb_image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    results = pose_model.process(rgb_image)
    label = "No person detected"
    
    if results.pose_landmarks:
        # Draw landmarks
        mp_drawing.draw_landmarks(img, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
        label = classify(results.pose_landmarks.landmark, st.session_state.activity_state)

    # Update label history for smoothing
    st.session_state.activity_state["labels"].append(label)
    
    # Show most common label
    if st.session_state.activity_state["labels"]:
        shown = max(
            set(st.session_state.activity_state["labels"]), 
            key=list(st.session_state.activity_state["labels"]).count
        )
    else:
        shown = label

    # Draw label on frame
    cv2.rectangle(img, (0, 0), (img.shape[1], 42), (0, 0, 0), -1)
    cv2.putText(
        img, f"Activity: {shown}", (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2
    )
    
    return av.VideoFrame.from_ndarray(img, format="bgr24")


# Initialize session state
if "activity_state" not in st.session_state:
    st.session_state.activity_state = {
        "ankle_gap": deque(maxlen=20),
        "labels": deque(maxlen=7),
    }

# Only show webrtc if model loaded successfully
if pose_model is not None:
    webrtc_streamer(
        key="live-har",
        video_frame_callback=video_frame_callback,
        media_stream_constraints={
            "video": {"width": 640, "height": 480},
            "audio": False
        },
        rtc_configuration={
            "iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]
        },
        async_processing=True,
    )

    st.caption(
        "💡 **Tips**: Use good lighting, stand 2–3 m from the camera, keep your full body visible. "
        "Walk sideways for best walking detection."
    )
else:
    st.error("Cannot start video stream. Please try one of the solutions above.")
