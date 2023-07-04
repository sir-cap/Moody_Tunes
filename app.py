import streamlit as st
import cv2
import numpy as np
import time
import os
import re
import pandas as pd
import spotipy
from spotipy.oauth2 import SpotifyOAuth
from tensorflow import keras
from keras.models import load_model
from keras.preprocessing.image import img_to_array

songs = pd.read_csv('cleaned_songs.csv')

# Path for the model
face_classifier = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
classifier = load_model('model.h5')

emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']

# Adding a countdown
countdown_time = 3  # Set the countdown time in seconds
countdown_start = False  # Flag to indicate if countdown has started
countdown_end_time = None  # Variable to store the countdown end time
detected_emotion = None  # Variable to store the detected emotion


def get_recommendations(emotion, songs):
    emotion_songs = songs[songs['Mood'].str.lower() == emotion.lower()]

    if not emotion_songs.empty:
        recommended_songs = emotion_songs.sample(n=7)
        return recommended_songs
    else:
        return pd.DataFrame()


def moody_tunes(folder_path, emotion, songs):
    # Get the list of files in the folder
    files = os.listdir(folder_path)

    # Sorting the files by their modified time in descending order
    sorted_files = sorted(files, key=lambda x: os.path.getmtime(os.path.join(folder_path, x)), reverse=True)

    if sorted_files:
        last_file = sorted_files[0]
        # Extract the emotion from the last added file name
        match = re.search(r'^(.*?)---', last_file)
        if match:
            detected_emotion = match.group(1).lower()

            if detected_emotion != emotion.lower():
                st.write("Please wait while we analyze your mood...")
                return

            # Filter songs by mood/emotion
            emotion_songs = songs[songs['Mood'].str.lower() == detected_emotion.lower()]

            if not emotion_songs.empty:
                # Randomly select 7 songs
                random_songs = emotion_songs.sample(n=7)
                st.markdown('<div style="display: flex; justify-content: center;">', unsafe_allow_html=True)
                st.dataframe(random_songs[['Track', 'Artist']])
                st.markdown('</div>', unsafe_allow_html=True)

        else:
            st.write("Try again, folks!")


def main():
    global countdown_start, countdown_end_time  # Mark variables as global
    st.title(":headphones: Moody Tunes :headphones:")
    st.subheader("Get song recommendations based on your face mood")

    # Create a button to start the mood detection
    check_mood_button = st.button("Let's capture your mood", help="Click here to check your mood")

    if check_mood_button:
        countdown_start = True
        countdown_end_time = time.time() + countdown_time

    if countdown_start:
        while countdown_start and time.time() < countdown_end_time:
            countdown_remaining = int(countdown_end_time - time.time()) + 1
            if countdown_remaining > 0:
                countdown_text = str(countdown_remaining)
                st.text(countdown_text)
            time.sleep(1)  # Add a small delay to avoid the continuous loop

        if countdown_start and time.time() >= countdown_end_time:
            countdown_start = False  # Reset the countdown

            cap = cv2.VideoCapture(0)
            ret, frame = cap.read()
            cap.release()

            if ret:
                labels = []
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                faces = face_classifier.detectMultiScale(gray)

                if len(faces) > 0:
                    (x, y, w, h) = faces[0]  # Consider the first detected face only
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 255), 2)
                    roi_gray = gray[y:y + h, x:x + w]
                    roi_gray = cv2.resize(roi_gray, (48, 48), interpolation=cv2.INTER_AREA)

                    if np.sum([roi_gray]) != 0:
                        roi = roi_gray.astype('float') / 255.0
                        roi = img_to_array(roi)
                        roi = np.expand_dims(roi, axis=0)

                        prediction = classifier.predict(roi)[0]
                        label = emotion_labels[prediction.argmax()]
                        label_position = (x, y - 11)
                        cv2.putText(frame, label, label_position, cv2.FONT_HERSHEY_DUPLEX, 1, (0, 0, 255), 2)

                        detected_emotion = label  # Store the detected emotion
                    else:
                        cv2.putText(frame, 'No Faces', (30, 80), cv2.FONT_HERSHEY_DUPLEX, 1, (0, 0, 255), 2)
                        detected_emotion = None
                else:
                    cv2.putText(frame, 'No Faces', (30, 80), cv2.FONT_HERSHEY_DUPLEX, 1, (0, 0, 255), 2)
                    detected_emotion = None

                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                st.write('Great job!')  # Display "Great job!" and the captured image
                st.image(frame_rgb, channels="RGB")
                if detected_emotion is not None:
                    # Save the captured image with emotion and timestamp
                    picture_folder = "pictures"
                    os.makedirs(picture_folder, exist_ok=True)
                    timestamp = time.strftime("%Y%m%d-%H%M%S")
                    picture_filename = f"{detected_emotion}---{timestamp}.jpg"
                    picture_path = os.path.join(picture_folder, picture_filename)
                    cv2.imwrite(picture_path, frame)

                    st.subheader(f"For your {detected_emotion} mood, your tunes are:")
                    songs_df = pd.read_csv('cleaned_songs.csv')  # Load songs data
                    moody_tunes(picture_folder, detected_emotion, songs_df)
                else:
                    st.warning("No faces detected. Please try again,folks!")


if __name__ == "__main__":
    main()