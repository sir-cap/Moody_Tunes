#importing libraries
import streamlit as st
from streamlit import secrets
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
import subprocess
from io import BytesIO

#adding passwords
spotipy_client_id = secrets["SPOTIPY_CLIENT_ID"]
spotipy_client_secret = secrets["SPOTIPY_CLIENT_SECRET"]
cloudinary_api_key = secrets["CLOUDINARY_API_KEY"]
cloudinary_api_secret = secrets["CLOUDINARY_API_SECRET"]
SPOTIFY_USER_ID = secrets["SPOTIFY_USER_ID"]
CLOUDINARY_CLOUD_NAME = secrets["CLOUDINARY_CLOUD_NAME"]

# Adding background — plain flat color (no gradient), cool blue + chocolate brown palette
# (#7CA7EB / #402924), matching the theme colors in .streamlit/config.toml.
page_bg = """
<style>
[data-testid="stAppViewContainer"] {
    background: #150d0b;
}
hr {
    background-color: rgba(124, 167, 235, 0.5) !important;
}
h1 {
    color: #7CA7EB !important;
}
[data-testid="stVerticalBlockBorderWrapper"] {
    border-color: rgba(124, 167, 235, 0.45) !important;
}
</style>
"""

# Initialize Cloudinary configuration
cloudinary.config(
    cloud_name=CLOUDINARY_CLOUD_NAME,
    api_key=cloudinary_api_key,
    api_secret=cloudinary_api_secret
)

# Load the logo once and render a responsive header (logo + title + optional subtitle).
# Replaces the old absolutely-positioned floating logo, which overlapped the title on narrow
# (mobile) screens, and the raw-HTML flex version that broke emoji-shortcode rendering
# (st.markdown's shortcode->emoji conversion doesn't run inside raw HTML). Native st.columns
# already stacks responsively on narrow screens, so no custom CSS/media queries needed.
_logo_path = "homepage_image.png"

def render_header(title, subtitle=None):
    logo_col, text_col = st.columns([1, 4], vertical_alignment="center")
    with logo_col:
        st.image(_logo_path, width=130)
    with text_col:
        st.title(title)
        if subtitle:
            st.caption(subtitle)

# Function to save the captured image on Cloudinary
def save_image_on_cloudinary(image_data, filename):
    # Convert the image data to bytes and create an in-memory file
    image_bytes = BytesIO()
    Image.fromarray(image_data).save(image_bytes, format='JPEG')
    image_bytes.seek(0)

    response = cloudinary.uploader.upload(image_bytes, public_id=filename)
    return response['secure_url']

# Function to treat the captured image and return of image with emotion
def detect_emotion(cv_image):
    # Define the variable detected_emotion before the if block
    detected_emotion = None

    gray = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)
    faces = face_classifier.detectMultiScale(gray, minNeighbors=2)

    if len(faces) > 0:
        (x, y, w, h) = faces[0]  # Consider the first detected face only
        cv_image_with_label = cv_image.copy()  # Create a copy of the original image
        cv2.rectangle(cv_image_with_label, (x, y), (x + w, y + h), (255, 0, 0), 4)
        roi_gray = gray[y:y + h, x:x + w]
        roi_gray = cv2.resize(roi_gray, (48, 48), interpolation=cv2.INTER_AREA)

        if np.sum([roi_gray]) != 0:
            roi = roi_gray.astype('float') / 255.0
            roi = img_to_array(roi)
            roi = np.expand_dims(roi, axis=0)

            prediction = classifier.predict(roi)[0]
            label = emotion_labels[prediction.argmax()]
            label_position = (x, y - 11)
            cv2.putText(cv_image_with_label, label, label_position, cv2.FONT_HERSHEY_DUPLEX, 3, (0, 255, 255), 2)

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
        recommended_songs = emotion_songs.sample(n=10)
        return recommended_songs
    else:
        return pd.DataFrame()

# Function to create a Spotify playlist using the recommended songs from the cleaned_songs df
def create_spotify_playlist(recommended_songs, username, emotion):
    # Create a new playlist
    playlist_name = f"MoodyTunes for a {emotion} day - {time.strftime('%d/%m/%Y')}"
    sp = spotipy.Spotify(auth_manager=spotipy.oauth2.SpotifyOAuth(scope="playlist-modify-public", client_id=spotipy_client_id,
                                                             client_secret=spotipy_client_secret,redirect_uri="https://moodytunes.streamlit.app/callback"))


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
    st.warning(f"🎧 Listen to your MoodyTunes on [Spotify]({playlist_url}) 🎧")

# Function for streamlit homepage structure and capture the image with emotion and return recommended songs playlist
def main():
    global detected_emotion  # Mark variables as global
    st.sidebar.title("Navigation")

    app_mode = st.sidebar.selectbox("Choose a page", ["Home", "About Moody Tunes"])

    if app_mode == "Home":
        st.markdown(page_bg, unsafe_allow_html=True)

        render_header("MOODY TUNES", "🎧 Get song recommendations based on your face mood")
        st.divider()

        upload_tab, camera_tab = st.tabs(["📁 Upload a photo", "📸 Take a photo"])
        with upload_tab:
            uploaded_file = st.file_uploader("", type=["jpg", "png", "jpeg"], label_visibility="collapsed")
        with camera_tab:
            camera_file = st.camera_input("", label_visibility="collapsed")

        # Whichever widget the user actually used wins (only one will be non-None at a time
        # in normal use, but prefer a fresh camera shot if somehow both are set).
        active_file = camera_file if camera_file is not None else uploaded_file

        detected_emotion = None  # Reset detected emotion to None

        if active_file is not None:
            # Convert the uploaded/captured file to an OpenCV image
            file_bytes = np.asarray(bytearray(active_file.read()), dtype=np.uint8)
            cv_image = cv2.imdecode(file_bytes, 1)  # 1 indicates loading the image in color

            # Perform emotion detection
            if cv_image is not None:
                with st.spinner("Detecting emotion from the photo..."):
                    countdown_time = 3  # Set the countdown time in seconds
                    countdown_start = True  # Flag to indicate if countdown has started
                    countdown_end_time = time.time() + countdown_time
                    detected_emotion, rgb_image = detect_emotion(cv_image)

                if detected_emotion is not None:
                    st.success('Great job! 👍')
                    # Save the image on Cloudinary
                    timestamp = time.strftime("%Y%m%d-%H%M%S")
                    picture_filename = f"{detected_emotion}---{timestamp}.jpg"
                    cloudinary_url = save_image_on_cloudinary(rgb_image, picture_filename)

                    with st.container(border=True):
                        # Display the uploaded and processed image
                        st.image(rgb_image, use_container_width=True)

                        # Create a container for the recommended songs and subheader
                        st.subheader(f"For your {detected_emotion} mood, your tunes are:")
                        songs_df = pd.read_csv('cleaned_songs.csv')  # Load songs dataframe
                        recommended_songs = get_recommendations(detected_emotion, songs_df)
                        if not recommended_songs.empty:
                            st.dataframe(recommended_songs[['Track', 'Artist']], use_container_width=True)
                            try:
                                create_spotify_playlist(recommended_songs, SPOTIFY_USER_ID, detected_emotion)
                            except spotipy.SpotifyBaseException:
                                st.info("Couldn't auto-create a Spotify playlist right now (the Spotify connection needs to be re-authorized), but here are your song recommendations above! 🎵")
                else:
                    detected_emotion = None
                    st.warning('No face detected in the photo. Try again! 😅')
            else:
                detected_emotion = None
                st.warning('Unable to read the photo. Please try again, folks!')

        # Adding about page and the homepage image
    elif app_mode == "About Moody Tunes":
        st.markdown(page_bg, unsafe_allow_html=True)
        render_header("About Moody Tunes")
        st.write("Moody Tunes is a web app that recognizes your mood using your facial expression and gives the user music suggestions from the same mood")
        st.divider()
        st.markdown("**How it works:**")
        st.write("1. Upload a photo, or take one with your camera, to start the mood detection.")
        st.write("2. The application will detect your facial expression and display it on the screen.")
        st.write("4. Based on your expression, the application will recommend songs that match your mood.")
        st.write("5. A Spotify page will be open for you to listen to your Moody Tunes.")
        st.divider()
        st.markdown("**Note:**")
        st.write("For the mood detection to work accurately, ensure that your face is well-illuminated and directly facing the camera.")
        st.warning("For more information, please reach out to diogoacapitao@gmail.com")
if __name__ == "__main__":
    main()