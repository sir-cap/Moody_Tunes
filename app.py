#Importing libraries
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
import base64

#Adding background
page_bg="""
<style>
[data-testid="stAppViewContainer"]
{
background: rgb(2,0,36);
background: linear-gradient(338deg, rgba(2,0,36,1) 0%, rgba(25,73,117,1) 79%, rgba(84,112,167,1) 99%)
}
</style>
"""

st.markdown(page_bg,unsafe_allow_html=True)


#Adding homepage image
homepage_image_path = "homepage_image.png"
homepage_image = open(homepage_image_path, "rb").read()
homepage_image_encoded = base64.b64encode(homepage_image).decode()
homepage_url = "http://localhost:8501"

st.markdown( # to costumize the logo on the page using HTML & CSS
    """
    <style>
    .image-container {
        position: absolute;
        top: q0px;
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

#Adding the songs dataframe and the page link for spotify playlist
songs = pd.read_csv('cleaned_songs.csv')
os.environ["http://localhost:8502/callback"] = "http://localhost:8502/callback"

# Path for the model
face_classifier = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
classifier = load_model('model.h5')

#defiying emotion clusters and the variable to save the detected emotion
emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']
detected_emotion = None  # Variable to store the detected emotion

# Adding a countdown
countdown_time = 3  # Set the countdown time in seconds
countdown_start = False  # Flag to indicate if countdown has started
countdown_end_time = None  # Variable to store the countdown end time
detected_emotion = None  # Variable to store the detected emotion

#Function to create a spotify playlist usinf the recommended songs from the cleaned_songs df
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

#Function to match the songs moods with the captured images mood
def get_recommendations(emotion, songs):
    emotion_songs = songs[songs['Mood'].str.lower() == emotion.lower()]

    if not emotion_songs.empty:
        recommended_songs = emotion_songs.sample(n=7)
        return recommended_songs
    else:
        return pd.DataFrame()

#Function to detect the emotions, save the image and get the name of the emotion
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
                st.markdown('</div>', unsafe_allow_html=True)

                create_spotify_playlist(recommended_songs, '1168069412', detected_emotion)

        else:
            st.write("Try again, folks!")


#function for streamlit homepage structure and capture the image with emotion and return recommended songs playlist
def main():
    global countdown_start, countdown_end_time  # Mark variables as global
    st.title(":headphones: Moody Tunes :headphones:")
    st.subheader("Get song recommendations based on your face mood")
    st.divider()

    # Create a button to start the mood detection
    check_mood_button = st.button("Let's capture your mood", help="Click here to start")
    cap = None  # Initialize the cap variable

    if check_mood_button:
        loading = st.empty()
        loading.write("Capturing your mood...")
        countdown_start = True
        countdown_end_time = time.time() + countdown_time

        # Initialize the camera
        cap = cv2.VideoCapture(0)

    if countdown_start:
        progress_bar = st.progress(0)
        while countdown_start and time.time() < countdown_end_time:
            countdown_remaining = countdown_end_time - time.time()
            progress = int(((countdown_time - countdown_remaining) / countdown_time) * 100)
            progress_bar.progress(progress)
            time.sleep(0.1)  # Add a small delay to avoid the continuous loop

        if countdown_start and time.time() >= countdown_end_time:
            countdown_start = False  # Reset the countdown
            progress_bar.empty()  # Remove the progress bar

            loading.empty()  # Clear the loading message

            if cap is not None:
                # Read a frame from the camera
                ret, frame = cap.read()

                if ret:
                    labels = []
                    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    faces = face_classifier.detectMultiScale(gray, minNeighbors=2)

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
                            st.success('Great job! :thumbsup:')
                        else:
                            detected_emotion = None
                            st.warning('Try again, folks! :pick:')

                        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
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
                            songs_df = pd.read_csv('cleaned_songs.csv')  # Load songs dataframe
                            moody_tunes(picture_folder, detected_emotion, songs_df)

    # Release the camera and clean up
    if cap is not None:
        cap.release()

if __name__ == "__main__":
    main()