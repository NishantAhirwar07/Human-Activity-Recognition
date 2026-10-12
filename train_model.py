"""
Train the Human Activity Recognition LSTM and save it as har_model.keras.

Usage:
    1. Download and extract the UCI HAR Dataset so the folder
       "UCI HAR Dataset" sits next to this file.
       https://archive.ics.uci.edu/dataset/240/human+activity+recognition+using+smartphones
    2. Run:  python train_model.py
       (or:  python train_model.py "path/to/UCI HAR Dataset")
    3. Upload the generated har_model.keras next to app.py in your GitHub repo.
"""
import os
import sys

import numpy as np
import tensorflow as tf
from sklearn.metrics import classification_report
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input
from tensorflow.keras.models import Sequential

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = sys.argv[1] if len(sys.argv) > 1 else os.path.join(BASE_DIR, "UCI HAR Dataset")
MODEL_PATH = os.path.join(BASE_DIR, "har_model.keras")

SIGNALS = [
    "body_acc_x", "body_acc_y", "body_acc_z",
    "total_acc_x", "total_acc_y", "total_acc_z",
    "body_gyro_x", "body_gyro_y", "body_gyro_z",
]
LABELS = [
    "Walking", "Walking Upstairs", "Walking Downstairs",
    "Sitting", "Standing", "Laying",
]


def load_sensor_data(folder):
    data = []
    for signal in SIGNALS:
        path = os.path.join(DATA_PATH, folder, "Inertial Signals", f"{signal}_{folder}.txt")
        data.append(np.loadtxt(path, dtype=np.float32))
    return np.stack(data, axis=2).astype(np.float32)


def load_labels(folder):
    return np.loadtxt(os.path.join(DATA_PATH, folder, f"y_{folder}.txt"), dtype=np.int32).astype(np.int32) - 1


if not os.path.isdir(DATA_PATH):
    sys.exit(f"Dataset folder not found: {DATA_PATH}")

X_train, X_test = load_sensor_data("train"), load_sensor_data("test")
y_train, y_test = load_labels("train"), load_labels("test")

print("Train:", X_train.shape, "Test:", X_test.shape)

# Standardize sensor values across all channels to improve LSTM convergence and training stability.
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train.reshape(-1, X_train.shape[-1])).reshape(X_train.shape).astype(np.float32)
X_test = scaler.transform(X_test.reshape(-1, X_test.shape[-1])).reshape(X_test.shape).astype(np.float32)

# Use a fixed validation split to avoid recomputing data at every epoch and improve training consistency.
val_split = int(len(X_train) * 0.1)
X_tr, X_val = X_train[:-val_split], X_train[-val_split:]
y_tr, y_val = y_train[:-val_split], y_train[-val_split:]

model = Sequential([
    Input(shape=(128, 9)),
    LSTM(64),
    Dropout(0.3),
    Dense(32, activation="relu"),
    Dense(6, activation="softmax"),
])
model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
model.summary()

callbacks = [
    EarlyStopping(monitor="val_loss", patience=3, restore_best_weights=True)
]

history = model.fit(
    X_tr,
    y_tr,
    epochs=10,
    batch_size=64,
    validation_data=(X_val, y_val),
    callbacks=callbacks,
    verbose=1,
)

loss, acc = model.evaluate(X_test, y_test, verbose=0)
print(f"\nTest loss: {loss:.4f} | Test accuracy: {acc * 100:.2f}%")
preds = np.argmax(model.predict(X_test, verbose=0), axis=1)
print(classification_report(y_test, preds, target_names=LABELS, zero_division=0))

model.save(MODEL_PATH)
print("Saved model to:", MODEL_PATH)
