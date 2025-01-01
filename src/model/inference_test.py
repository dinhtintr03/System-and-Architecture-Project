import pickle
import cv2
import mediapipe as mp
import numpy as np

# Load the pre-trained model
model_dict = pickle.load(open('src\\saved_weights\\model_final.p', 'rb'))
model = model_dict['model']

# Initialize webcam and MediaPipe Hands model
cap = cv2.VideoCapture(0)
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)

# Label dictionary
labels_dict = {
    0: 'A', 1: 'B', 2: 'C', 3: 'D', 4:'E', 5: 'F', 6: 'I', 7: 'K', 8: 'L', 9: 'O', 10: 'U', 11: 'V', 12: 'W', 13: 'Y' 
}

while True:
    data_aux = []  
    x_ = []
    y_ = []

    ret, frame = cap.read()
    H, W, _ = frame.shape

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(frame_rgb)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                frame, hand_landmarks, mp_hands.HAND_CONNECTIONS,
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style())

            # Collect landmark coordinates
            for i in range(len(hand_landmarks.landmark)):
                x = hand_landmarks.landmark[i].x
                y = hand_landmarks.landmark[i].y
                x_.append(x)
                y_.append(y)

        # Normalize landmarks and prepare for prediction
        for i in range(len(hand_landmarks.landmark)):
            x = hand_landmarks.landmark[i].x
            y = hand_landmarks.landmark[i].y
            data_aux.append((x - min(x_)) / (max(x_) - min(x_)))
            data_aux.append((y - min(y_)) / (max(y_) - min(y_)))

    # Check if at least one hand is detected
    if data_aux:
        x1 = int(min(x_) * W) - 10
        y1 = int(min(y_) * H) - 10
        x2 = int(max(x_) * W) - 10
        y2 = int(max(y_) * H) - 10

        # Prediction and confidence score
        prediction = model.predict([np.asarray(data_aux)])
        confidence_scores = model.predict_proba([np.asarray(data_aux)])

        # Get the predicted character and its confidence score
        predicted_label = int(prediction[0])
        predicted_character = labels_dict[predicted_label]
        confidence_score = confidence_scores[0][predicted_label] * 100  # Convert to percentage

        # Display results
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 0), 4)
        cv2.putText(
            frame, f'{predicted_character} ({confidence_score:.2f}%)', 
            (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 0, 0), 3, cv2.LINE_AA
        )

    cv2.imshow('frame', frame)
    cv2.waitKey(1)

cap.release()
cv2.destroyAllWindows()
