import pickle

import cv2
import mediapipe as mp
import numpy as np

handmark_model_path = './model/hand_landmarker.task'

model_dict = pickle.load(open('./model/image_classifier.p', 'rb'))
model = model_dict['model']

cap = cv2.VideoCapture(0)

BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions

options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=handmark_model_path), num_hands=2)

labels_dict = {0: 'A', 1: 'B', 2: 'C'}

while True:

    data_aux = []

    ret, frame = cap.read()

    x_ = []
    y_ = []
    H, W, _ = frame.shape

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
    
    results = HandLandmarker.create_from_options(options).detect(frame_img)

    if results.hand_landmarks:
        for hand in results.hand_landmarks:
            for landmark in hand:
                x = landmark.x
                y = landmark.y

                x_.append(x)
                y_.append(y)
                
                data_aux.append(x)
                data_aux.append(y)

        x_min = int(min(x_ * W) - 10)
        y_min = int(min(y_) * H) - 10
        x_max = int(max(x_) * W) + 10
        y_max = int(max(y_) * H) + 10
        
        prediction = model.predict([np.asarray(data_aux)])

        predicted_character = labels_dict[int(prediction[0])]

        cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 0, 0), 4)
        cv2.putText(frame, predicted_character, (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 0, 0), 3,
                    cv2.LINE_AA)

    cv2.imshow('frame', frame)
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()