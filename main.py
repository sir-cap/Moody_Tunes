#importing libraries
from tensorflow import keras
from keras.models import load_model
from time import sleep
from keras.preprocessing.image import img_to_array
from keras.preprocessing import image
import cv2
import numpy as np
import datetime 
import time 
import pandas as pd
import os 
import re

#path for the model
face_classifier = cv2.CascadeClassifier(r'/Users/diogocapitao/Documents/DA_Bootcamp/Project/final_project/haarcascade_frontalface_default.xml')
classifier = load_model(r'/Users/diogocapitao/Documents/DA_Bootcamp/Project/final_project/model.h5')

emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']

cap = cv2.VideoCapture(0)

#adding a countdown
countdown_time = 3  # Set the countdown time in seconds
countdown_start = False  # Flag to indicate if countdown has started
countdown_end_time = None  # Variable to store the countdown end time
detected_emotion = None  # Variable to store the detected emotion

#loop to make the camera works and add emotions - face recognition
while True:
    _, frame = cap.read()
    if frame is None:
        continue
    labels = []
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_classifier.detectMultiScale(gray)

    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 255), 2)
        roi_gray = gray[y:y + h, x:x + w]
        roi_gray = cv2.resize(roi_gray, (48, 48), interpolation=cv2.INTER_AREA)

        if np.sum([roi_gray]) != 0:
            roi = roi_gray.astype('float') / 255.0
            roi = img_to_array(roi)
            roi = np.expand_dims(roi, axis=0)

            prediction = classifier.predict(roi)[0]
            label = emotion_labels[prediction.argmax()]
            label_position = (x, y-11)
            cv2.putText(frame, label, label_position, cv2.FONT_HERSHEY_DUPLEX, 1, (0, 0, 255), 2)

            detected_emotion = label  # Store the detected emotion
        else:
            cv2.putText(frame, 'No Faces', (30, 80), cv2.FONT_HERSHEY_DUPLEX, 1, (0, 0, 255), 2)
    
    #adding countdown on screen
    if countdown_start and time.time() < countdown_end_time:
        countdown_remaining = int(countdown_end_time - time.time()) + 1
        if countdown_remaining > 0:
            countdown_text = str(countdown_remaining)
        else:
            countdown_text = 'Great job!'
        countdown_text_size = cv2.getTextSize(countdown_text, cv2.FONT_HERSHEY_DUPLEX, 2, 2)[0]
        text_x = int((frame.shape[1] - countdown_text_size[0]) / 2 - countdown_text_size[0] // 2)   # Calculate x coordinate for center alignment
        cv2.putText(frame, countdown_text, (text_x, 75), cv2.FONT_HERSHEY_DUPLEX, 2, (0, 255, 255), 2)

    cv2.imshow('Moody Tunes', frame)

    key = cv2.waitKey(1)
    if key & 0xFF == ord('m'):
        if not countdown_start:
            countdown_start = True
            countdown_end_time = time.time() + countdown_time

    elif key & 0xFF == ord('t'):
        break

    if countdown_start and time.time() >= countdown_end_time:
        timestamp = datetime.datetime.now().strftime('%Y%m%d%H%M%S')
        file_name = f'{detected_emotion}---{timestamp}.jpg'  # Add detected emotion to the file name
        file_path = r'/Users/diogocapitao/Documents/DA_Bootcamp/Project/final_project/pictures/' + file_name #picture saving
        cv2.imwrite(file_path, frame)

        countdown_start = False  # Reset the countdown
        cv2.waitKey(600) # add a break between caputre and last message "Great job"
        cv2.putText(frame, 'Great job!', (text_x, 50), cv2.FONT_HERSHEY_DUPLEX, 2, (0, 255, 255), 2)
        cv2.imshow('Moody Tunes', frame)
        cv2.waitKey(1300)  # Display "Great job!" and the captured image for 1.3 seconds

cap.release()
cv2.destroyAllWindows()
