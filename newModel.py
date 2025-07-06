import cv2
import time
import numpy as np
from tensorflow.keras.models import load_model

# Load pretrained model
model = load_model("keras_model.h5")

# You may need to update these based on the model's training order
# Try this order first
classes = ['rock', 'paper', 'scissors']

# Image preprocessing function
def preprocess_image(frame):
    img = cv2.resize(frame, (224, 224))          # Resize for this model
    img = img.astype("float32") / 255.0           # Normalize
    img = np.expand_dims(img, axis=0)             # Add batch dimension
    return img

# Initialize webcam (change to 1 if 0 doesn't work)
cap = cv2.VideoCapture(0)

font = cv2.FONT_HERSHEY_SIMPLEX
countdown = 3
font_scale = 5
thickness = 8
indicator_frames = 30
indicator_counter = 0
screenshot_taken = False
start_time = None
start_countdown = False

while True:
    ret, frame = cap.read()
    if not ret:
        break

    if not start_countdown:
        start_time = time.time()
        start_countdown = True

    elapsed = int(time.time() - start_time)

    if elapsed < countdown:
        # Show countdown numbers
        text = str(countdown - elapsed)
        (w, h), _ = cv2.getTextSize(text, font, font_scale, thickness)
        x = (frame.shape[1] - w) // 2
        y = (frame.shape[0] + h) // 2
        cv2.putText(frame, text, (x, y), font, font_scale, (0, 0, 255), thickness)

    elif elapsed == countdown and not screenshot_taken:
        frame_captured = frame.copy()
        screenshot_taken = True
        indicator_counter = 30

        # Preprocess and predict
        processed = preprocess_image(frame_captured)
        predictions = model.predict(processed)

        predicted_index = np.argmax(predictions[0])
        confidence = predictions[0][predicted_index]
        prediction_text = f"{classes[predicted_index]} ({confidence:.2f})"

        print("Raw predictions:", predictions[0])
        print(f"Prediction: {prediction_text}")

    elif screenshot_taken:
        if indicator_counter > 0:
            cv2.rectangle(frame, (0, 0), (frame.shape[1]-1, frame.shape[0]-1), (0, 0, 255), 10)
            indicator_counter -= 1
            cv2.putText(frame, prediction_text, (50, 80), font, 2, (255, 255, 0), 4)
        else:
            cv2.putText(frame, "Press 'r' to restart or 'q' to quit", (30, frame.shape[0] - 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

    cv2.imshow("Rock Paper Scissors", frame)
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    if key == ord('r'):
        screenshot_taken = False
        start_countdown = False
        indicator_counter = 0

cap.release()
cv2.destroyAllWindows()