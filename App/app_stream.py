
import joblib
import cv2
import mediapipe as mp
import numpy as np
import time

handmark_model_path = './model/hand_landmarker.task'
model = joblib.load('./model/image_classifier.joblib')

labels_dict = {0: 'A', 1: 'B', 2: 'C'}

cap = cv2.VideoCapture(0)

BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
HandLandmarkerResult = mp.tasks.vision.HandLandmarkerResult
VisionRunningMode = mp.tasks.vision.RunningMode

def print_result(result: HandLandmarkerResult, output_image: mp.Image, timestamp_ms: int):
    print('hand landmarker result: {}'.format(result))

options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=handmark_model_path),
    running_mode=VisionRunningMode.LIVE_STREAM,
    num_hands=1,
    min_hand_detection_confidence=0.7,
    min_hand_presence_confidence=0.7,
    min_tracking_confidence=0.7,
    result_callback=print_result)

while cap.isOpened():
    
    ret, frame = cap.read()
    if ret is False:
        break
    
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
    
    with HandLandmarker.create_from_options(options) as landmarker:
        H, W, _ = frame.shape
        data_aux = []
        frame_timestamp_ms = int(time.time() * 3000)
        results = landmarker.detect_async(mp_image, frame_timestamp_ms)

        if results:
            for hand in results.hand_landmarks:
                for landmark in hand:
                    x = landmark.x
                    y = landmark.y
                    
                    data_aux.append(x)
                    data_aux.append(y)

            x_min = int(min(data_aux[::2]) * W) - 10
            y_min = int(min(data_aux[1::2]) * H) - 10
            x_max = int(max(data_aux[::2]) * W) + 10
            y_max = int(max(data_aux[1::2]) * H) + 10
            
            cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (128, 0, 255), 4)
            
            prediction = model.predict([np.asarray(data_aux)])
            predicted_character = labels_dict[int(prediction)]
            
            cv2.putText(frame, predicted_character, (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            
        cv2.imshow('frame', frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()