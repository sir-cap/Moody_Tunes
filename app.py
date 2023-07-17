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

def get_camera_image():
    # Call the st.camera_input() function with the 'label' argument
    camera_image = st.camera_input(label="Press the button 'let's capture your mood' to start")
    return camera_image

def capture_image_with_emotion(emotion, timeframe):
    if camera_image is not None:
        # Save the captured image to Cloudinary
        image_bytes = camera_image.read()
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        picture_filename = f"{emotion}_{timeframe}_{timestamp}.jpg"
        cloudinary_url = save_uploaded_image_on_cloudinary(image_bytes, picture_filename)

        # Display the uploaded image
        image_np_array = np.frombuffer(image_bytes, np.uint8)
        frame = cv2.imdecode(image_np_array, cv2.IMREAD_COLOR)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        st.image(frame_rgb, caption=f"Detected Emotion: {emotion}", use_column_width=True)

captured_image = None

# Function to save the captured image on Cloudinary
def save_captured_image(image, emotion, timestamp):
    # Save the captured image to Cloudinary with emotion and timestamp as the filename
    picture_filename = f"{emotion}_{timestamp}.jpg"
    image_bytes = cv2.imencode(".jpg", image)[1].tobytes()
    cloudinary_url = save_uploaded_image_on_cloudinary(image_bytes, picture_filename)
    return cloudinary_url


def capture_image():
    # Check if the camera input widget has already been created
    if 'camera_image' not in st.session_state:
        # Create the camera input widget if it hasn't been created
        st.session_state.camera_image = st.camera_input(label="Press the button 'Let's capture your mood' to start")
    
    return st.session_state.camera_image

# Adding the songs dataframe and the page link for Spotify playlist
songs = pd.read_csv('cleaned_songs.csv')
os.environ["http://localhost:8501/callback"] = "http://localhost:8502/callback"

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

# Function to create a Spotify playlist using the recommended songs from the cleaned_songs df
def create_spotify_playlist(recommended_songs, username, emotion):
    # Create a new playlist
    playlist_name = f"MoodyTunes for a {emotion} day - {time.strftime('%d/%m/%Y')}"
    client_id = os.environ.get("SPOTIFY_CLIENT_ID")
    client_secret = os.environ.get("SPOTIFY_CLIENT_SECRET")
    redirect_uri = os.environ.get("SPOTIFY_REDIRECT_URI")
    sp = spotipy.Spotify(auth_manager=SpotifyOAuth(scope="playlist-modify-public", client_id=client_id,
                                                   client_secret=client_secret, redirect_uri=redirect_uri))

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

# Function to match the songs moods with the captured images mood
def get_recommendations(emotion, songs):
    emotion_songs = songs[songs['Mood'].str.lower() == emotion.lower()]

    if not emotion_songs.empty:
        recommended_songs = emotion_songs.sample(n=7)
        return recommended_songs
    else:
        return pd.DataFrame()

# Function to detect the emotions, save the image and get the name of the emotion
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

            # Filter songs by mood/emotion
            emotion_songs = songs[songs['Mood'].str.lower() == detected_emotion.lower()]

            if not emotion_songs.empty:
                # Randomly select 7 songs
                recommended_songs = emotion_songs.sample(n=7)
                st.markdown('<div style="display: flex; justify-content: center;">', unsafe_allow_html=True)
                st.dataframe(recommended_songs[['Track', 'Artist']])

                create_spotify_playlist(recommended_songs, '1168069412', detected_emotion)

        else:
            st.write("Try again, folks!")

#convers camera image to array

# Function for streamlit homepage structure and capture the image with emotion and return recommended songs playlist
def main():
    global countdown_start, countdown_end_time  # Mark variables as global
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

        # Get the camera image
        camera_image = st.camera_input(label="Press the button 'Let's capture your mood' to start")

        # Create a button to start the mood detection
        check_mood_button = st.button("Let's capture your mood", help="Click here to start")
        st.markdown('</div>', unsafe_allow_html=True)

        if camera_image is not None:
            global detected_emotion  # Use the global variable
            if check_mood_button:
                # Convert the camera image to a NumPy array using PIL and OpenCV
                camera_pil_image = Image.fromarray(camera_image.astype('uint8'), 'RGB')
                camera_np_array = pil_to_np_array(camera_pil_image)

                # Perform mood detection and song recommendation based on the uploaded image
                gray = cv2.cvtColor(camera_np_array, cv2.COLOR_BGR2GRAY)
                faces = face_classifier.detectMultiScale(gray, minNeighbors=2)

                if len(faces) > 0:
                    (x, y, w, h) = faces[0]  # Consider the first detected face only
                    cv2.rectangle(camera_np_array, (x, y), (x + w, y + h), (0, 255, 255), 2)
                    roi_gray = gray[y:y + h, x:x + w]
                    roi_gray = cv2.resize(roi_gray, (48, 48), interpolation=cv2.INTER_AREA)

                    if np.sum([roi_gray]) != 0:
                        roi = roi_gray.astype('float') / 255.0
                        roi = img_to_array(roi)
                        roi = np.expand_dims(roi, axis=0)

                        prediction = classifier.predict(roi)[0]
                        label = emotion_labels[prediction.argmax()]
                        label_position = (x, y - 11)
                        cv2.putText(camera_np_array, label, label_position, cv2.FONT_HERSHEY_DUPLEX, 1, (0, 0, 255), 2)

                        detected_emotion = label  # Store the detected emotion
                        frame_rgb = cv2.cvtColor(camera_np_array, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB

                        if detected_emotion is not None:
                            st.success('Great job! :thumbsup:')
                            st.image(frame_rgb, caption=f"Detected Emotion: {detected_emotion}", use_column_width=True)

                            # Save the captured image to Cloudinary and display it
                            timestamp = time.strftime("%Y%m%d-%H%M%S")
                            cloudinary_url = save_captured_image(camera_np_array, detected_emotion, timestamp)
                            st.image(cloudinary_url, caption=f"Detected Emotion: {detected_emotion}", use_column_width=True)

                            # Perform mood detection and song recommendation based on the uploaded image
                            moody_tunes(detected_emotion, songs)

                        else:
                            detected_emotion = None
                            st.warning('Try again, folks! :pick:')

                else:
                    detected_emotion = None
                    st.warning('Try again, folks! :pick:')

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
        st.write("1. Click on the 'Let's capture your mood' button to start the mood detection.")
        st.write("2. The application will access your device's camera and capture a frame.")
        st.write("3. It will detect your facial expression and display it on the screen.")
        st.write("4. Based on your expression, the application will recommend songs that match your mood.")
        st.write("5. You can listen to the recommended songs on Spotify.")
        st.divider()
        st.markdown("**Note:**")
        st.write("For the mood detection to work accurately, ensure that your face is well-illuminated and directly facing the camera.")
        st.warning("For more information, please reach out to diogo.capitao.576@gmail.com")


if __name__ == "__main__":
    main()
