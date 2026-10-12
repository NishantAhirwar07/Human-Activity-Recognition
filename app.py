from collections import deque
import os
import sys
import tempfile
import warnings

# Suppress warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# === Set up MediaPipe cache BEFORE any imports ===
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

# Import mediapipe with proper error handling
try:
    import mediapipe as mp
    mp_pose = mp.solutions.pose
    mp_drawing = mp.solutions.drawing_utils
    L = mp_pose.PoseLandmark
except (ImportError, AttributeError) as e:
    st.error(f"❌ MediaPipe import failed: {e}")
    st.info(
        "Please ensure MediaPipe is properly installed:\n"
        "```bash\n"
        "pip install --upgrade mediapipe\n"
        "```"
    )
    st.stop()

from streamlit_webrtc import webrtc_streamer

pose_model = None
model_error = None


@st.cache_resource
def initialize_pose_model():
    """Initialize MediaPipe Pose model with error handling."""
    try:
        model = mp_pose.Pose(
            model_complexity=0,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
            static_image_mode=False,
        )
        return model, None
    
    except PermissionError as e:
        error_msg = (
            "🔒 **Permission Denied**: MediaPipe cannot access model files.\n\n"
            "This is a **Streamlit Cloud limitation**. The app needs a platform "
            "with writable filesystem access."
        )
        return None, error_msg
    
    except Exception as e:
        error_msg = f"❌ Error initializing MediaPipe: {type(e).__name__}: {str(e)[:100]}"
        return None, error_msg


# Initialize model
pose_model, model_error = initialize_pose_model()

# Display error and solutions
if model_error:
    st.warning(model_error)
    with st.expander("✅ How to Fix (3 Options)", expanded=True):
        st.markdown("""
        ### Option 1: Run Locally (Fastest & Guaranteed to Work)
        ```bash
        git clone https://github.com/NishantAhirwar07/Human-Activity-Recognition.git
        cd Human-Activity-Recognition
        pip install -r requirements.txt
        streamlit run app.py
        ```
        Then open **http://localhost:8501** in your browser.
        
        ---
        
        ### Option 2: Deploy on Render (Free)
        1. Push this repo to GitHub
        2. Go to [render.com](https://render.com)
        3. Click "New +" → "Web Service"
        4. Connect your GitHub repo
        5. Set Runtime: **Python 3.12**
        6. Build command: `pip install -r requirements.txt`
        7. Start command: `streamlit run app.py --server.port=10000`
        8. Add env var: `STREAMLIT_SERVER_PORT=10000`
        
        ---
        
        ### Option 3: Deploy on Railway (Free Credit)
        1. Go to [railway.app](https://railway.app)
        2. Click "New Project" → "Deploy from GitHub"
        3. Select your repo
        4. Railway auto-detects Streamlit
        5. Deploy automatically
        
        ---
        
        **Why Streamlit Cloud doesn't work?**
        - Streamlit Cloud's sandbox has read-only package directories
        - MediaPipe needs writable access to load model files
        - This is a platform limitation, not a code bug
        """)
    
    st.error("❌ Cannot start app without MediaPipe model. Please use one of the solutions above.")
    st.stop()


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

# Start video stream
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
