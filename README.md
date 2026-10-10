# 🏃 Human Activity Recognition using LSTM

A deep learning project that classifies human physical activities from smartphone sensor data (accelerometer and gyroscope) using an **LSTM (Long Short-Term Memory)** neural network built with TensorFlow/Keras.

---

## 📌 Overview

Human Activity Recognition (HAR) identifies what a person is doing based on motion sensor signals. This project trains an LSTM model on raw inertial signals from the **UCI HAR Dataset** to recognize six everyday activities:

| Label | Activity |
|-------|----------|
| 0 | Walking |
| 1 | Walking Upstairs |
| 2 | Walking Downstairs |
| 3 | Sitting |
| 4 | Standing |
| 5 | Laying |

---

## 📊 Dataset

- **Name:** [UCI Human Activity Recognition Using Smartphones](https://archive.ics.uci.edu/dataset/240/human+activity+recognition+using+smartphones)
- **Source:** Waist-mounted smartphone (accelerometer + gyroscope), sampled at 50 Hz
- **Window size:** 128 readings per sample (2.56 seconds)
- **Input signals (9 channels):**
  - `body_acc` (x, y, z)
  - `total_acc` (x, y, z)
  - `body_gyro` (x, y, z)
- **Split:** 7,352 training samples / 2,947 test samples

Input shape per sample: `(128, 9)`

> ⚠️ The dataset is not included in this repository. Download it from the link above and keep the `UCI HAR Dataset` folder available (see [Usage](#-usage)).

---

## 🧠 Model Architecture

```
Input (128 timesteps × 9 signals)
        │
   LSTM (64 units)
        │
   Dropout (0.3)
        │
   Dense (32, ReLU)
        │
   Dense (6, Softmax)
```

- **Total parameters:** 21,222
- **Optimizer:** Adam
- **Loss:** Sparse Categorical Crossentropy
- **Epochs:** 10
- **Batch size:** 128
- **Validation split:** 10%

---

## 📈 Results

**Test accuracy: 80.62%** (Test loss: 0.5516)

| Activity | Precision | Recall | F1-Score |
|----------|-----------|--------|----------|
| Walking | 0.75 | 0.94 | 0.83 |
| Walking Upstairs | 0.78 | 0.66 | 0.72 |
| Walking Downstairs | 0.67 | 0.72 | 0.69 |
| Sitting | 0.83 | 0.74 | 0.78 |
| Standing | 0.82 | 0.83 | 0.83 |
| Laying | 0.98 | 0.92 | 0.95 |
| **Overall (macro avg)** | **0.81** | **0.80** | **0.80** |

The model performs best on **Laying** (clearly distinct sensor orientation) and finds **Walking Upstairs/Downstairs** and **Sitting/Standing** harder to separate, which is common for this dataset.

---

## ⚙️ Requirements

- Python 3.9+
- TensorFlow 2.x
- NumPy
- Matplotlib
- Seaborn
- scikit-learn

Install dependencies:

```bash
pip install tensorflow numpy matplotlib seaborn scikit-learn
```

---

## 🚀 Usage

### Option 1: Google Colab (recommended)

1. Open `Human_Activity_Recognition_.ipynb` in [Google Colab](https://colab.research.google.com/).
2. Run the first cell and upload `UCI HAR Dataset.zip` when prompted.
3. Run all remaining cells to train, evaluate, and save the model.

### Option 2: Run locally

1. Download and extract the UCI HAR Dataset.
2. In the notebook, remove the `google.colab` upload/download code and set:
   ```python
   DATA_PATH = 'path/to/UCI HAR Dataset'
   ```
3. Run the notebook with Jupyter.

### Load the trained model

```python
import numpy as np
from tensorflow.keras.models import load_model

model = load_model('har_model.keras')

labels = ['Walking', 'Walking Upstairs', 'Walking Downstairs',
          'Sitting', 'Standing', 'Laying']

# sample shape must be (1, 128, 9)
prediction = model.predict(sample)
print("Predicted activity:", labels[np.argmax(prediction)])
```

---

## 📁 Project Structure

```
├── Human_Activity_Recognition_.ipynb   # Training & evaluation notebook
├── har_model.keras                     # Trained model (saved after running the notebook)
└── README.md
```

---

## 🔮 Future Improvements

- Train for more epochs and add early stopping
- Try stacked LSTMs, Bidirectional LSTM, or CNN-LSTM hybrids
- Normalize/standardize input signals
- Hyperparameter tuning (units, dropout, learning rate)
- Deploy as a real-time activity tracker or web app

---

## 🙏 Acknowledgements

Dataset: Anguita, D., Ghio, A., Oneto, L., Parra, X., & Reyes-Ortiz, J. L. (2013). *A Public Domain Dataset for Human Activity Recognition Using Smartphones.* ESANN 2013.

---

## 📄 License

This project is licensed under the MIT License. Feel free to use and modify it.
