import os
import numpy as np
import librosa

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.utils import to_categorical

# Emotion mapping for RAVDESS
emotion_dict = {
    "01": "neutral",
    "02": "calm",
    "03": "happy",
    "04": "sad",
    "05": "angry",
    "06": "fearful",
    "07": "disgust",
    "08": "surprised"
}

# MFCC feature extraction
def extract_features(file_path):
    audio, sr = librosa.load(file_path, duration=3, offset=0.5)

    mfcc = librosa.feature.mfcc(
        y=audio,
        sr=sr,
        n_mfcc=40
    )

    return np.mean(mfcc.T, axis=0)

# Dataset folder
dataset_path = "dataset"

X = []
y = []

# Load audio files
for root, dirs, files in os.walk(dataset_path):

    for file in files:

        if file.endswith(".wav"):

            try:
                emotion_code = file.split("-")[2]

                feature = extract_features(
                    os.path.join(root, file)
                )

                X.append(feature)
                y.append(
                    emotion_dict[emotion_code]
                )

            except Exception as e:
                print("Skipped:", file)

print("Total samples loaded:", len(X))

# Convert to arrays
X = np.array(X)

encoder = LabelEncoder()
y = encoder.fit_transform(y)

y = to_categorical(y)

# Reshape for LSTM
X = X.reshape(X.shape[0], X.shape[1], 1)

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# Build LSTM Model
model = Sequential()

model.add(
    LSTM(
        128,
        input_shape=(40, 1)
    )
)

model.add(Dropout(0.3))

model.add(
    Dense(
        64,
        activation="relu"
    )
)

model.add(Dropout(0.3))

model.add(
    Dense(
        y.shape[1],
        activation="softmax"
    )
)

# Compile
model.compile(
    optimizer="adam",
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

# Train
history = model.fit(
    X_train,
    y_train,
    validation_data=(X_test, y_test),
    epochs=20,
    batch_size=32
)

# Evaluate
loss, accuracy = model.evaluate(
    X_test,
    y_test
)

print("\nFinal Accuracy:", round(accuracy * 100, 2), "%")

# Predict a few samples
predictions = model.predict(X_test[:5])

for i in range(5):
    predicted = encoder.inverse_transform(
        [np.argmax(predictions[i])]
    )[0]

    actual = encoder.inverse_transform(
        [np.argmax(y_test[i])]
    )[0]

    print(
        f"Actual: {actual} | Predicted: {predicted}"
    )