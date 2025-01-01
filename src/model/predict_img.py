import cv2 
import mediapipe as mp
import pickle
import numpy as np
import os

# Load the pre-trained model
model_dict = pickle.load(open('src\\saved_weights\\model_final.p', 'rb'))
model = model_dict['model']

# Initialize MediaPipe Hands model
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)
labels_dict = {0: 'A', 1: 'B', 2: 'C'}

# Directory containing images for prediction
IMAGE_DIR = 'data\\processed_data\\1\\2.jpg'

for img_name in os.listdir(IMAGE_DIR):
    # Read each image
    img_path = os.path.join(IMAGE_DIR, img_name)
    img = cv2.imread(img_path)
    if img is None:
        print(f"Could not read {img_path}. Skipping.")
        continue
    
    H, W, _ = img.shape
    data_aux = []
    x_ = []
    y_ = []

    # Convert image to RGB as required by MediaPipe
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(img_rgb)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                img,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS,
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style()
            )

            # Collect landmarks for prediction
            for i in range(len(hand_landmarks.landmark)):
                x = hand_landmarks.landmark[i].x
                y = hand_landmarks.landmark[i].y
                x_.append(x)
                y_.append(y)
                data_aux.append(x)
                data_aux.append(y)

        # Bounding box around the detected hand
        x1 = int(min(x_) * W)
        y1 = int(min(y_) * H)
        x2 = int(max(x_) * W)
        y2 = int(max(y_) * H)

        # Make prediction
        prediction = model.predict([np.asarray(data_aux)])
        predicted_character = labels_dict[int(prediction[0])]
        
        # Display prediction on image
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 0), 4)
        cv2.putText(img, predicted_character, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 0, 0), 3, cv2.LINE_AA)

        # Show the image with predictions
        cv2.imshow('Prediction', img)
        cv2.waitKey(0)  # Press any key to move to the next image

cv2.destroyAllWindows()
