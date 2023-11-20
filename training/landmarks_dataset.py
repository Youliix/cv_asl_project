import json
import os

import mediapipe as mp
import cv2

handmark_model_path = './model/hand_landmarker.task'
DATA_DIR = './data'

BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions

options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=handmark_model_path), num_hands=2)

data = []
labels = []
scoring = []

for dir_entry in os.scandir(DATA_DIR):
    if dir_entry.is_dir():
        for img_entry in os.scandir(dir_entry.path):
            data_aux = []
            
            img = cv2.imread(img_entry.path)
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=img_rgb)
            
            hand_landmarker = HandLandmarker.create_from_options(options)
            results = hand_landmarker.detect(mp_img)
            for content in results.handedness:
                scoring.append(content[0].score)

            if results.hand_landmarks and len(results.hand_landmarks) == 1:
                for hand in results.hand_landmarks:
                    for landmark in hand:
                        x = landmark.x
                        y = landmark.y
                        # z = landmark.z
                        data_aux.append(x)
                        data_aux.append(y)
                        # data_aux.append(z)
                        
            data.append(data_aux)
            labels.append(dir_entry.name)

with open('./dataset/data.json', 'w') as f:
    json.dump({'data': data, 'labels': labels}, f)