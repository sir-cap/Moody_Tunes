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
from PIL import Image
import cloudinary
import cloudinary.uploader
import cloudinary.api
import base64
import subprocess
from io import BytesIO

# Adding background & asking for camera permission
page_bg = """
<style>
[data-testid="stAppViewContainer"] {
    background: rgb(2, 0, 36);
    background: radial-gradient(circle, rgba(2, 0, 36, 1) 0%, rgba(8, 8, 109, 1) 60%, rgba(0, 84, 255, 1) 80%);
}

#camera-container {
    position: relative;
    width: 100%;
    height: 0;
    padding-bottom: 56.25%;
}

#camera-container video {
    position: absolute;
    top: 0;
    left: 0;
    width: 100%;
    height: 100%;
    object-fit: cover;
}
</style>
"""

# Adding styles
primaryColor = "#c9c9c9"
backgroundColor = "#04044c"
secondaryBackgroundColor = "#04044c"
textColor = "#c9c9c9"

# Initialize Cloudinary configuration
cloudinary.config(
    cloud_name="dpylcpsoo",
    api_key="874926159578349",
    api_secret="phLxggqDlqsgWFpVwTwLk15Hw88"
)

# Function to save the captured image on Cloudinary
def save_image_on_cloudinary(image_data, filename):
    # Convert the image data to bytes and create an in-memory file
    image_bytes = BytesIO()
    Image.fromarray(image_data).save(image_bytes, format='JPEG')
    image_bytes.seek(0)

    response = cloudinary.uploader.upload(image_bytes, public_id=filename)
    return response['secure_url']

def detect_emotion(cv_image):
    # Define the variable detected_emotion before the if block
    detected_emotion = None

    gray = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)
    faces = face_classifier.detectMultiScale(gray, minNeighbors=2)

    if len(faces) > 0:
        (x, y, w, h) = faces[0]  # Consider the first detected face only
        cv_image_with_label = cv_image.copy()  # Create a copy of the original image
        cv2.rectangle(cv_image_with_label, (x, y), (x + w, y + h), (255, 0, 0), 5)
        roi_gray = gray[y:y + h, x:x + w]
        roi_gray = cv2.resize(roi_gray, (48, 48), interpolation=cv2.INTER_AREA)

        if np.sum([roi_gray]) != 0:
            roi = roi_gray.astype('float') / 255.0
            roi = img_to_array(roi)
            roi = np.expand_dims(roi, axis=0)

            prediction = classifier.predict(roi)[0]
            label = emotion_labels[prediction.argmax()]
            label_position = (x, y - 11)
            cv2.putText(cv_image_with_label, label, label_position, cv2.FONT_HERSHEY_DUPLEX, 3, (0, 255, 255), 3)

            # Assign the detected_emotion inside the if block
            detected_emotion = label
        else:
            detected_emotion = None

        # Convert the modified image to RGB before returning
        rgb_image = cv2.cvtColor(cv_image_with_label, cv2.COLOR_BGR2RGB)
        

        # Return the detected emotion and the modified image
        return detected_emotion, rgb_image

    else:
        # Convert the original image to RGB before returning
        rgb_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)

        # Return the detected emotion and the original image (since no face detected)
        return detected_emotion, rgb_image

# Adding the songs dataframe and the page link for Spotify playlist
songs = pd.read_csv('cleaned_songs.csv')
os.environ["http://localhost:8501/callback"] = "https://moodytunes.streamlit.app/callback"

# Path for the model
face_classifier = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
classifier = load_model('model.h5')

# Defining emotion clusters and the variable to save the detected emotion
emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']
detected_emotion = None  # Variable to store the detected emotion

# Adding a countdown
countdown_time = 3  # Set the countdown time in seconds
countdown_start = False  # Flag to indicate if countdown has started
countdown_end_time = None  # Variable to store the countdown end time
detected_emotion = None  # Variable to store the detected emotion

# Function to match the songs moods with the captured images mood
def get_recommendations(emotion, songs):
    emotion_songs = songs[songs['Mood'].str.lower() == emotion.lower()]

    if not emotion_songs.empty:
        recommended_songs = emotion_songs.sample(n=7)
        return recommended_songs
    else:
        return pd.DataFrame()

# Function to create a Spotify playlist using the recommended songs from the cleaned_songs df
def create_spotify_playlist(recommended_songs, username, emotion):
    # Create a new playlist
    playlist_name = f"MoodyTunes for a {emotion} day - {time.strftime('%d/%m/%Y')}"
    sp = spotipy.Spotify(auth_manager=SpotifyOAuth(scope="playlist-modify-public", client_id="e5557a3edfe0413f9babf4fefe546e02",
                                                   client_secret="aa8ebb963acc4ffbb79ae4ed396d61ec", redirect_uri="https://moodytunes.streamlit.app/callback"))

    playlist = sp.user_playlist_create(user=username, name=playlist_name, public=True)
    playlist_id = playlist['id']

    # Search for each track and artist and add them to the playlist
    for index, row in recommended_songs.iterrows():
        track_name = row['Track']
        artist_name = row['Artist']

        # Search for the track and artist on Spotify
        search_query = f"track:{track_name} artist:{artist_name}"
        result = sp.search(q=search_query, type='track', limit=1)

        if result['tracks']['items']:
            track_uri = result['tracks']['items'][0]['uri']
            sp.playlist_add_items(playlist_id, [track_uri])

    # Add info if it's successful
    playlist_url = playlist['external_urls']['spotify']
    st.subheader(f"Listen to your Moody Tunes on [Spotify]({playlist_url})")

# Function for streamlit homepage structure and capture the image with emotion and return recommended songs playlist
def main():
    global detected_emotion  # Mark variables as global
    st.sidebar.title("Navigation")

    app_mode = st.sidebar.selectbox("Choose a page", ["Home", "About Moody Tunes"])

    if app_mode == "Home":
        st.markdown(page_bg, unsafe_allow_html=True)

        # Adding homepage image
        homepage_image_path = "homepage_image.png"
        homepage_image = open(homepage_image_path, "rb").read()
        homepage_image_encoded = base64.b64encode(homepage_image).decode()
        homepage_url = "http://localhost:8501"

        st.markdown(
            """
            <style>
            .image-container {
                position: absolute;
                top: 0px;
                right: 10px;
                z-index: 9999;
            }
            </style>
            """,
            unsafe_allow_html=True,
        )
        st.markdown(
            f"""
            <div class="image-container">
                <a href="#" onclick="window.location.reload(); return false;">
                    <img src="data:image/png;base64,{homepage_image_encoded}" alt="Homepage" width="100" height="100">
                </a>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.title("MOODY TUNES")
        st.subheader(":headphones: Get song recommendations based on your face mood")
        st.divider()

        uploaded_file = st.file_uploader("",type=["jpg", "png", "jpeg"],label_visibility="visible")

        detected_emotion = None  # Reset detected emotion to None

        if uploaded_file is not None:
            # Convert the uploaded file to an OpenCV image
            file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
            cv_image = cv2.imdecode(file_bytes, 1)  # 1 indicates loading the image in color

            # Perform emotion detection
            if cv_image is not None:
                with st.spinner("Detecting emotion from the uploaded image..."):
                    countdown_time = 3  # Set the countdown time in seconds
                    countdown_start = True  # Flag to indicate if countdown has started
                    countdown_end_time = time.time() + countdown_time
                    detected_emotion, rgb_image = detect_emotion(cv_image) 

                if detected_emotion is not None:
                    st.success('Great job! :thumbsup:')
                    # Save the image on Cloudinary
                    timestamp = time.strftime("%Y%m%d-%H%M%S")
                    picture_filename = f"{detected_emotion}---{timestamp}.jpg"
                    cloudinary_url = save_image_on_cloudinary(cv_image, picture_filename)

                    # Display the uploaded and processed image
                    st.image(rgb_image, caption=f"(Emotion: {detected_emotion})", use_column_width=True)

                    # Create a container for the recommended songs and subheader
                    st.subheader(f"For your {detected_emotion} mood, your tunes are:")
                    songs_df = pd.read_csv('cleaned_songs.csv')  # Load songs dataframe
                    recommended_songs = get_recommendations(detected_emotion, songs_df)
                    if not recommended_songs.empty:
                        st.dataframe(recommended_songs[['Track', 'Artist']])
                        create_spotify_playlist(recommended_songs, '1168069412', detected_emotion)
                else:
                    detected_emotion = None
                    st.warning('No face detected in the uploaded image. Try again! :pick:')
            else:
                detected_emotion = None
                st.warning('Unable to read the uploaded image. Please try again.')
            
        # Adding about page and the homepage image 
    elif app_mode == "About Moody Tunes":
        st.markdown(page_bg, unsafe_allow_html=True) 
        homepage_image_path = "homepage_image.png"
        homepage_image = open(homepage_image_path, "rb").read()
        homepage_image_encoded = base64.b64encode(homepage_image).decode()
        homepage_url = "http://localhost:8501"
        st.markdown(
            """
            <style>
            .image-container {
                position: absolute;
                top: 0px;
                right: 10px;
                z-index: 9999;
            }
            </style>
            """,
            unsafe_allow_html=True,
        )
        st.markdown(
            f"""
            <div class="image-container">
                <a href="#" onclick="window.location.reload(); return false;">
                    <img src="data:image/png;base64,{homepage_image_encoded}" alt="Homepage" width="80" height="80">
                </a>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.title("About Moody Tunes")
        st.write("Moody Tunes is a user interface that recognizes your mood using your facial expression and gives the user music suggestions from the same mood")
        st.divider()
        st.markdown("**How it works:**")
        st.write("1. Drag and drop or click in 'Browse files' to upload your picture to start the mood detection.")
        st.write("2. The application will detect your facial expression and display it on the screen.")
        st.write("4. Based on your expression, the application will recommend songs that match your mood.")
        st.write("5. You can listen to the recommended songs on Spotify directly using the link presented.")
        st.divider()
        st.markdown("**Note:**")
        st.write("For the mood detection to work accurately, ensure that your face is well-illuminated and directly facing the camera.")
        st.warning("For more information, please reach out to diogo.capitao.576@gmail.com")
if __name__ == "__main__":
    main()