import os

import numpy as np
import pandas as pd
import streamlit as st
import tensorflow as tf

# ----------------------------------------------------------------------
# Page setup
# ----------------------------------------------------------------------
st.set_page_config(
    page_title="Human Activity Recognition",
    page_icon="🏃",
    layout="centered",
)

MODEL_PATH = "har_model.keras"
DATASET_PATH = "UCI HAR Dataset"  # optional, only used for the demo tab

LABELS = [
    "Walking",
    "Walking Upstairs",
    "Walking Downstairs",
    "Sitting",
    "Standing",
    "Laying",
]

EMOJIS = ["🚶", "🪜⬆️", "🪜⬇️", "🪑", "🧍", "🛌"]

SIGNALS = [
    "body_acc_x", "body_acc_y", "body_acc_z",
    "total_acc_x", "total_acc_y", "total_acc_z",
    "body_gyro_x", "body_gyro_y", "body_gyro_z",
]

WINDOW = 128  # readings per sample (2.56 s at 50 Hz)


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------
@st.cache_resource
def load_model():
    return tf.keras.models.load_model(MODEL_PATH)


@st.cache_data
def load_demo_samples():
    """Load the UCI HAR test set (only if the dataset folder is present)."""
    data = []
    for signal in SIGNALS:
        path = os.path.join(
            DATASET_PATH, "test", "Inertial Signals", f"{signal}_test.txt"
        )
        data.append(np.loadtxt(path))
    X = np.stack(data, axis=2)
    y = np.loadtxt(os.path.join(DATASET_PATH, "test", "y_test.txt")).astype(int) - 1
    return X, y


def parse_csv(file):
    """Read an uploaded CSV into an array of shape (n_windows, 128, 9)."""
    df = pd.read_csv(file, header=None)
    df = df.apply(pd.to_numeric, errors="coerce")

    # Drop a header row if the first row is non-numeric
    if df.iloc[0].isna().all():
        df = df.iloc[1:]
    df = df.dropna(how="all").reset_index(drop=True)

    values = df.to_numpy(dtype=np.float32)

    # Accept (9 x N) files by transposing
    if values.shape[1] != 9 and values.shape[0] == 9:
        values = values.T

    if values.shape[1] != 9:
        raise ValueError(
            f"Expected 9 columns (signals), but found {values.shape[1]}."
        )
    if np.isnan(values).any():
        raise ValueError("The file contains missing or non-numeric values.")
    if values.shape[0] < WINDOW:
        raise ValueError(
            f"Need at least {WINDOW} rows (readings), but found {values.shape[0]}."
        )

    n_windows = values.shape[0] // WINDOW
    return values[: n_windows * WINDOW].reshape(n_windows, WINDOW, 9)


def show_prediction(model, window):
    """Predict a single window of shape (128, 9) and display the result."""
    probs = model.predict(window[np.newaxis, ...], verbose=0)[0]
    idx = int(np.argmax(probs))

    st.success(f"### {EMOJIS[idx]} {LABELS[idx]}")
    st.metric("Confidence", f"{probs[idx] * 100:.2f}%")

    st.markdown("**Probability for each activity**")
    prob_df = pd.DataFrame({"Activity": LABELS, "Probability": probs}).set_index(
        "Activity"
    )
    st.bar_chart(prob_df)

    with st.expander("View sensor signals"):
        signal_df = pd.DataFrame(window, columns=SIGNALS)
        st.line_chart(signal_df)

    return idx


# ----------------------------------------------------------------------
# UI
# ----------------------------------------------------------------------
st.title("🏃 Human Activity Recognition")
st.write(
    "Predict a person's activity from smartphone accelerometer and gyroscope "
    "data using an **LSTM** model trained on the UCI HAR dataset."
)

if not os.path.exists(MODEL_PATH):
    st.error(
        f"Model file `{MODEL_PATH}` not found. Place it in the same folder as "
        "`app.py` (run the notebook to generate it)."
    )
    st.stop()

model = load_model()

tab_upload, tab_demo, tab_about = st.tabs(["📂 Upload CSV", "🧪 Demo sample", "ℹ️ About"])

# ---------------------------- Upload tab ------------------------------
with tab_upload:
    st.markdown(
        f"""
Upload a CSV file with **9 columns** (one per signal, in this order) and at
least **{WINDOW} rows** (readings):

`{", ".join(SIGNALS)}`

If the file has more rows, it is split into windows of {WINDOW} readings and
each window is classified.
"""
    )

    uploaded = st.file_uploader("Choose a CSV file", type=["csv"])

    if uploaded is not None:
        try:
            windows = parse_csv(uploaded)
        except Exception as e:
            st.error(f"Could not read the file: {e}")
        else:
            st.info(f"Found **{len(windows)}** window(s) of {WINDOW} readings.")

            if len(windows) == 1:
                show_prediction(model, windows[0])
            else:
                probs = model.predict(windows, verbose=0)
                preds = np.argmax(probs, axis=1)

                result_df = pd.DataFrame(
                    {
                        "Window": np.arange(1, len(windows) + 1),
                        "Activity": [LABELS[i] for i in preds],
                        "Confidence (%)": (np.max(probs, axis=1) * 100).round(2),
                    }
                )
                st.dataframe(result_df, use_container_width=True, hide_index=True)

                st.markdown("**Activity distribution**")
                counts = (
                    pd.Series([LABELS[i] for i in preds])
                    .value_counts()
                    .reindex(LABELS, fill_value=0)
                )
                st.bar_chart(counts)

                st.markdown("**Inspect a window**")
                chosen = st.slider("Window number", 1, len(windows), 1)
                show_prediction(model, windows[chosen - 1])

# ----------------------------- Demo tab -------------------------------
with tab_demo:
    if not os.path.isdir(DATASET_PATH):
        st.info(
            f"To use this tab, place the extracted `{DATASET_PATH}` folder next "
            "to `app.py`. Otherwise, use the **Upload CSV** tab."
        )
    else:
        X_test, y_test = load_demo_samples()
        st.write(f"Pick a sample from the test set ({len(X_test)} samples).")

        index = st.number_input(
            "Sample number", min_value=0, max_value=len(X_test) - 1, value=0, step=1
        )

        if st.button("Predict", type="primary"):
            pred = show_prediction(model, X_test[int(index)])
            actual = int(y_test[int(index)])
            if pred == actual:
                st.caption(f"✅ Actual activity: {LABELS[actual]}")
            else:
                st.caption(f"❌ Actual activity: {LABELS[actual]}")

# ----------------------------- About tab ------------------------------
with tab_about:
    st.markdown(
        """
**Model:** LSTM (64) → Dropout (0.3) → Dense (32, ReLU) → Dense (6, Softmax)

**Dataset:** UCI Human Activity Recognition Using Smartphones

**Test accuracy:** ~80.6%

**Activities:** Walking, Walking Upstairs, Walking Downstairs, Sitting,
Standing, Laying
"""
    )
