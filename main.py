import cv2
import random
import time
import os
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout

def train_and_save_model():
    data_dir = "rps-cv-images"
    img_size = (128, 128)
    batch_size = 32
    datagen = ImageDataGenerator(
        rescale=1.0/255,
        validation_split=0.2
    )

    train_gen = datagen.flow_from_directory(
        data_dir,
        target_size=img_size,
        batch_size=batch_size,
        class_mode='categorical',
        subset='training',
        shuffle=True
    )

    val_gen = datagen.flow_from_directory(
        data_dir,
        target_size=img_size,
        batch_size=batch_size,
        class_mode='categorical',
        subset='validation',
        shuffle=False
    )

    class_indices = train_gen.class_indices
    class_names = {v: k for k, v in class_indices.items()}

    with open("class_mapping.txt", "w") as f:
        for i in range(len(class_names)):
            f.write(f"{i}: {class_names[i]}\n")

    print("Class indices mapping:", class_indices)

    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=(128, 128, 3)),
        MaxPooling2D(pool_size=(2, 2)),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D(pool_size=(2, 2)),
        Conv2D(128, (3, 3), activation='relu'),
        MaxPooling2D(pool_size=(2, 2)),
        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(len(class_indices), activation='softmax')
    ])

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    model.summary()

    model.fit(train_gen, epochs=5, validation_data=val_gen)
    model.save("rps_model.h5")
    return class_names

def preprocess_image(frame):
    img = cv2.resize(frame, (128, 128))
    img = img.astype("float32") / 255.0
    img = np.expand_dims(img, axis=0)
    return img

def get_computer_choice():
    return random.choice(['rock', 'paper', 'scissors'])

def determine_winner(user_choice, computer_choice):
    if user_choice == computer_choice:
        return "Tie!"
    if (user_choice == 'rock' and computer_choice == 'scissors') or \
       (user_choice == 'paper' and computer_choice == 'rock') or \
       (user_choice == 'scissors' and computer_choice == 'paper'):
        return "You win!"
    return "Computer wins!"

def apply_green_screen_mask(frame):
    # Convert to HSV
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Define green color range (tweak if needed)
    lower_green = np.array([35, 40, 40])
    upper_green = np.array([85, 255, 255])

    # Create a mask where green is detected
    mask = cv2.inRange(hsv, lower_green, upper_green)
    mask_inv = cv2.bitwise_not(mask)

    # Create a uniform green background
    green_bg = np.full_like(frame, (0, 255, 0))  # solid green

    # Combine green bg with masked hand area
    fg = cv2.bitwise_and(frame, frame, mask=mask_inv)
    bg = cv2.bitwise_and(green_bg, green_bg, mask=mask)
    combined = cv2.add(fg, bg)

    return combined

if __name__ == "__main__":
    class_names = train_and_save_model()
    model = load_model("rps_model.h5")

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
    game_result = None
    computer_choice = None
    confidence_threshold = 0.7

    user_choices = {'rock': 0, 'paper': 0, 'scissors': 0}
    games_played = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        display_frame = frame.copy()

        if not start_countdown:
            cv2.putText(display_frame, "Press SPACE to start", (20, 40), font, 0.8, (0, 255, 0), 2)
            cv2.putText(display_frame, "Press Q to quit", (20, 80), font, 0.8, (0, 255, 0), 2)
        else:
            elapsed = int(time.time() - start_time)
            if elapsed < countdown:
                text = str(countdown - elapsed)
                (w, h), _ = cv2.getTextSize(text, font, font_scale, thickness)
                x = (display_frame.shape[1] - w) // 2
                y = (display_frame.shape[0] + h) // 2
                cv2.putText(display_frame, text, (x, y), font, font_scale, (0, 0, 255), thickness)
                cv2.putText(display_frame, "Show your move!", (50, 50), font, 1, (255, 255, 0), 2)

            elif elapsed == countdown and not screenshot_taken:
                frame_captured = apply_green_screen_mask(frame.copy())
                screenshot_taken = True
                indicator_counter = 30

                # Save the original webcam image
                cv2.imwrite("original_captured_frame.png", frame_captured)
                cv2.imshow("Original Screenshot", frame_captured)

                processed = preprocess_image(frame_captured)

                # Save the preprocessed image (resized and normalized)
                img_to_save = (processed[0] * 255).astype('uint8')
                cv2.imwrite("processed_captured_image.png", img_to_save)
                cv2.imshow("Processed Input", img_to_save)

                predictions = model.predict(processed)
                predicted_index = np.argmax(predictions[0])
                confidence = predictions[0][predicted_index]
                print("Raw prediction:", predictions[0])

                if confidence >= confidence_threshold:
                    user_choice = class_names[predicted_index]
                    user_choices[user_choice] += 1
                    game_result = determine_winner(user_choice, get_computer_choice())
                    games_played += 1
                else:
                    user_choice = "unclear"
                    game_result = "Couldn't detect your move clearly"

                print(f"Prediction: {user_choice} ({confidence:.2f}) | Result: {game_result}")

            elif screenshot_taken:
                if indicator_counter > 0:
                    cv2.rectangle(display_frame, (0, 0), (display_frame.shape[1]-1, display_frame.shape[0]-1), (0, 0, 255), 10)
                    indicator_counter -= 1

                if user_choice != "unclear":
                    cv2.putText(display_frame, f"You: {user_choice} ({confidence:.2f})", (30, 100), font, 1, (0, 255, 0), 2)
                    cv2.putText(display_frame, game_result, (30, 150), font, 1.5, (0, 0, 255), 3)
                else:
                    cv2.putText(display_frame, "Move not recognized", (30, 50), font, 1, (0, 0, 255), 2)
                    cv2.putText(display_frame, "Try again", (30, 100), font, 1, (0, 0, 255), 2)

                cv2.putText(display_frame, "Press SPACE to play again", (30, 200), font, 0.8, (255, 255, 0), 2)

        cv2.imshow("Rock Paper Scissors Game", display_frame)
        key = cv2.waitKey(1) & 0xFF

        if key == ord('q'):
            break
        elif key == ord(' '):
            if not start_countdown or screenshot_taken:
                start_time = time.time()
                start_countdown = True
                screenshot_taken = False
                game_result = None
                user_choice = None
                computer_choice = None

    cap.release()
    cv2.destroyAllWindows()