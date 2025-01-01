import pickle
import cv2
import mediapipe as mp
import numpy as np

# Load the trained model
model_dict = pickle.load(open('src\\saved_weights\\model_final.p', 'rb'))
model = model_dict['model']

# Initialize webcam capture
cap = cv2.VideoCapture(0)

# Setup MediaPipe hands
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

# Initialize MediaPipe hands model
hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)

# Labels mapping from numeric index to characters
labels_dict = {
    0: 'A', 1: 'B', 2: 'C', 3: 'D', 4:'E', 5: 'F', 6: 'I', 7: 'K', 
    8: 'L', 9: 'O', 10: 'U', 11: 'V', 12: 'W', 13: 'Y' 
}

while True:
    data_aux = []  
    x_ = []
    y_ = []

    ret, frame = cap.read()

    H, W, _ = frame.shape

    # Convert the frame to RGB as Mediapipe requires it
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process the frame and detect hands
    results = hands.process(frame_rgb)

    # Check if hands are detected and process the first hand
    if results.multi_hand_landmarks:
        # Only process the first detected hand
        hand_landmarks = results.multi_hand_landmarks[0]
        
        # Draw landmarks and connections on the frame
        mp_drawing.draw_landmarks(
            frame,  # image to draw
            hand_landmarks,  # model output
            mp_hands.HAND_CONNECTIONS,  # hand connections
            mp_drawing_styles.get_default_hand_landmarks_style(),
            mp_drawing_styles.get_default_hand_connections_style())

        # Extract the x, y coordinates of the hand landmarks
        for i in range(len(hand_landmarks.landmark)):
            x = hand_landmarks.landmark[i].x
            y = hand_landmarks.landmark[i].y
            x_.append(x)
            y_.append(y)

        # Normalize the hand landmarks
        for i in range(len(hand_landmarks.landmark)):
            x = hand_landmarks.landmark[i].x
            y = hand_landmarks.landmark[i].y
            data_aux.append((x - min(x_)) / (max(x_) - min(x_)))
            data_aux.append((y - min(y_)) / (max(y_) - min(y_)))

        # If data is available, predict the letter
        if data_aux:
            # Define bounding box based on landmarks
            x1 = int(min(x_) * W) - 10
            y1 = int(min(y_) * H) - 10
            x2 = int(max(x_) * W) - 10
            y2 = int(max(y_) * H) - 10

            # Predict the character using the trained model
            prediction = model.predict([np.asarray(data_aux)])
            predicted_character = labels_dict[int(prediction[0])]

            # Draw bounding box and predicted character on the frame
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 0), 4)
            cv2.putText(frame, predicted_character, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 0, 0), 3, 
                        cv2.LINE_AA)

    # Display the frame with the prediction
    cv2.imshow('frame', frame)
    
    # Exit if 'ESC' is pressed
    if cv2.waitKey(1) & 0xFF == 27:
        break

# Release the webcam and close all windows
cap.release()
cv2.destroyAllWindows()
