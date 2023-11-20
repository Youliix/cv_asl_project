import os
import logging

import numpy as np
import json
from joblib import dump

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

with open('./dataset/data.json', 'r') as f:
    data_dict = json.load(f)

new_data_dict = {'data': [], 'labels': []}
for index, data in enumerate(data_dict['data']):
    if len(data) != 0:
        new_data_dict['data'].append(data)
        new_data_dict['labels'].append(data_dict['labels'][index])

data = np.asarray(new_data_dict['data'])
labels = np.asarray(new_data_dict['labels'])

x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, shuffle=True, stratify=labels)

model = RandomForestClassifier()

model.fit(x_train, y_train)

y_predict = model.predict(x_test)

score = accuracy_score(y_predict, y_test)

logging.warning('{}% of samples were classified correctly !'.format(score * 100))

print(classification_report(y_test, y_predict))

dump(model, './model/image_classifier.joblib')
