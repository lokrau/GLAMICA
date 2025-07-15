# GET STOVE STATE
# Function to get the border of the stove
def get_stove_border_frame(stove_bbox, border_ratio=0.2):
    x1, y1, x2, y2 = stove_bbox
    width = x2 - x1
    height = y2 - y1

    bw = int(width * border_ratio)
    bh = int(height * border_ratio)

    outer = (x1, y1, x2, y2)
    inner = (x1 + bw, y1 + bh, x2 - bw, y2 - bh)

    return outer, inner

# Function to get the overlap of another bounding box with the stove frame
def overlaps_with_stove_frame(hand_box, outer, inner):
    def area(box):
        x1, y1, x2, y2 = box
        return max(0, x2 - x1) * max(0, y2 - y1)

    def intersection(boxA, boxB):
        xa1, ya1, xa2, ya2 = boxA
        xb1, yb1, xb2, yb2 = boxB
        x_left = max(xa1, xb1)
        y_top = max(ya1, yb1)
        x_right = min(xa2, xb2)
        y_bottom = min(ya2, yb2)

        if x_right < x_left or y_bottom < y_top:
            return 0
        return (x_right - x_left) * (y_bottom - y_top)

    hand_area = area(hand_box)
    i_outer = intersection(hand_box, outer)
    i_inner = intersection(hand_box, inner)

    frame_overlap = i_outer - i_inner
    return frame_overlap > 0.0

# GET FAUCET STATE
## Imports
import numpy as np
import sounddevice as sd
import librosa
import pickle
import tensorflow as tf
import tensorflow_hub as hub
from tensorflow.keras.models import load_model 

## Variables
SAMPLE_RATE = 16000
DURATION = 2  # seconds
YAMNET_MODEL_URL = "https://tfhub.dev/google/yamnet/1"

## Load the YAMNet model
yamnet_model = hub.load(YAMNET_MODEL_URL)
model = load_model("yamnet_faucet_recognition/model/yamnet_faucet_model.h5")
with open("yamnet_faucet_recognition/model/yamnet_label_encoder.pkl", "rb") as f:
    label_encoder = pickle.load(f)

## Functions
### Record audio
def record_audio(duration=DURATION, fs=SAMPLE_RATE):
    print(f"Recording for {duration}s...")
    audio = sd.rec(int(duration * fs), samplerate=fs, channels=1)
    sd.wait()
    return audio.flatten().astype(np.float32)

### Extract features with yamnet
def extract_yamnet_features(audio):
    # YAMNet expects float32 waveform at 16kHz
    scores, embeddings, spectrogram = yamnet_model(audio)
    return np.mean(embeddings.numpy(), axis=0)  # Shape: (1024,)

### Prediction function
def live_predict():
    audio = record_audio()
    features = extract_yamnet_features(audio)
    features = np.expand_dims(features, axis=0)  # Shape: (1, 1024)
    prediction = model.predict(features)
    predicted_index = np.argmax(prediction)
    predicted_label = label_encoder.inverse_transform([predicted_index])[0]
    print(f"Detected: {predicted_label}\n")
    return predicted_label


