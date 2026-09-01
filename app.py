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
# Refresh token from a one-time manual OAuth authorization (see reference/spotify-reauth.md).
# Stored in Secrets, not a local .cache file, so it survives Streamlit Cloud redeploys.
SPOTIFY_REFRESH_TOKEN = secrets["SPOTIFY_REFRESH_TOKEN"]

# Adding background — plain flat color (no gradient). Inverted from the earlier version:
# cool blue (#93BAF2) background now, chocolate brown (#402924) text/accent color, matching
# the theme colors in .streamlit/config.toml. Every text color here is picked to read clearly
# on the light blue background — no white-on-light text anywhere.
page_bg = """
<style>
[data-testid="stAppViewContainer"] {
    background: #93BAF2;
}
hr {
    background-color: rgba(46, 27, 22, 0.55) !important;
}
h1, h2, h3, label, [data-testid="stWidgetLabel"] p {
    color: #2E1B16 !important;
}
/* A lighter brown for secondary accents (subtitle, tab labels) — gives some tonal variety
   against the darker brown used for headings/labels, instead of a single flat brown. */
[data-testid="stCaptionContainer"], [data-testid="stTab"] p {
    color: #4A3020 !important;
}
[data-testid="stVerticalBlockBorderWrapper"] {
    border-color: rgba(46, 27, 22, 0.5) !important;
}
a {
    color: #2E1B16 !important;
    font-weight: 600;
}
/* Keep the header's logo+title columns side by side at every screen size. Streamlit's
   columns stack on narrow viewports by default (each stColumn gets a forced
   min-width: calc(100% - 24px), so on mobile the logo ends up above the title instead of
   next to it). This app only uses st.columns for the header, so it's safe to override
   everywhere rather than scoping to a media query. */
[data-testid="stHorizontalBlock"] {
    flex-wrap: nowrap !important;
}
[data-testid="stColumn"] {
    min-width: 0 !important;
    width: auto !important;
}
/* Streamlit's default page padding (96px top / 160px bottom) left a lot of empty space
   above the header and below the last widget. */
[data-testid="stMainBlockContainer"] {
    padding-top: 52px !important;
    padding-bottom: 40px !important;
}
/* Shrink the header logo/title on narrow screens — at the fixed 130px logo width the header
   was taking up a big share of a phone screen's height. Desktop is untouched. This also caps
   the detected-photo image, but that one already fits its container, so it's a no-op there. */
@media (max-width: 640px) {
    [data-testid="stImage"] img {
        max-width: 70px !important;
        height: auto !important;
    }
    h1 {
        font-size: 1.8rem !important;
    }
    [data-testid="stCaptionContainer"] {
        font-size: 0.85rem !important;
    }
}
</style>
"""

# Initialize Cloudinary configuration
cloudinary.config(
    cloud_name=CLOUDINARY_CLOUD_NAME,
    api_key=cloudinary_api_key,
    api_secret=cloudinary_api_secret
)

# Brand-colored replacement for st.success/st.info/st.warning: Streamlit's built-in alert
# colors (green/blue/yellow) aren't part of this app's blue+brown palette and clash with it.
# html param lets a caller embed a styled link (see the Spotify box) instead of plain text.
def styled_message(html):
    st.markdown(
        f'<div style="background:#2E1B16;color:#EDE3DA;padding:12px 18px;'
        f'border-radius:8px;margin:16px 0;font-size:15px;">{html}</div>',
        unsafe_allow_html=True,
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

# Function to treat the captured image and return the image with emotion + confidence scores.
# Face detection uses OpenCV's YuNet (a small DNN, see face_detection_yunet.onnx), which is
# meaningfully more reliable than the Haar Cascade this used to use — especially on angled
# faces, partial occlusion, and varied lighting.
def detect_emotion(cv_image):
    detected_emotion = None
    prediction = None

    h, w = cv_image.shape[:2]
    face_detector.setInputSize((w, h))
    _, faces = face_detector.detect(cv_image)

    if faces is not None and len(faces) > 0:
        # Consider the most confident detection only. YuNet's box can extend slightly outside
        # the image (e.g. a small negative x/y near an edge) — clamp the start coordinates so
        # the slice below doesn't wrap around via negative indexing.
        x, y, box_w, box_h = faces[0][:4].astype(int)
        x, y = max(x, 0), max(y, 0)

        cv_image_with_label = cv_image.copy()  # Create a copy of the original image
        cv2.rectangle(cv_image_with_label, (x, y), (x + box_w, y + box_h), (255, 0, 0), 4)
        gray = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)
        roi_gray = gray[y:y + box_h, x:x + box_w]
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

        # Return the detected emotion, the modified image, and the full per-emotion confidence
        # scores (used to render a confidence chart in main()).
        return detected_emotion, rgb_image, prediction

    else:
        # Convert the original image to RGB before returning
        rgb_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)

        # Return the detected emotion and the original image (since no face detected)
        return detected_emotion, rgb_image, prediction

# Adding the songs dataframe and the page link for Spotify playlist
songs = pd.read_csv('cleaned_songs.csv')
os.environ["http://localhost:8501/callback"] = "https://moodytunes.streamlit.app/callback"

# Path for the model
face_detector = cv2.FaceDetectorYN.create(
    "face_detection_yunet.onnx", "", (320, 320), score_threshold=0.6
)
classifier = load_model('model.h5')

# Defining emotion clusters
emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']

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
    # Seed the auth manager with the stored refresh token via a MemoryCacheHandler instead of
    # relying on a local .cache file: Streamlit Cloud's filesystem resets on every redeploy, so
    # a file-based cache silently loses the token each time the app rebuilds. expires_at=0
    # forces an immediate refresh using SPOTIFY_REFRESH_TOKEN on first use.
    cache_handler = spotipy.cache_handler.MemoryCacheHandler(token_info={
        "refresh_token": SPOTIFY_REFRESH_TOKEN,
        "access_token": "",
        "expires_at": 0,
        "scope": "playlist-modify-public",
        "token_type": "Bearer",
    })
    sp = spotipy.Spotify(auth_manager=spotipy.oauth2.SpotifyOAuth(
        scope="playlist-modify-public",
        client_id=spotipy_client_id,
        client_secret=spotipy_client_secret,
        redirect_uri="https://moodytunes.streamlit.app/callback",
        cache_handler=cache_handler,
    ))


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
    # style="...!important" on the link: the global `a { color: ... !important }` rule (added
    # for other stray blue links) would otherwise beat this inline color too.
    styled_message(
        ':material/headphones: Listen to your MoodyTunes on '
        f'<a href="{playlist_url}" style="color:#93BAF2 !important;font-weight:700;">Spotify</a> :material/headphones:'
    )

# Function for streamlit homepage structure and capture the image with emotion and return recommended songs playlist
def main():
    st.sidebar.title("Navigation")

    app_mode = st.sidebar.selectbox("Choose a page", ["Home", "About Moody Tunes"])

    if app_mode == "Home":
        st.markdown(page_bg, unsafe_allow_html=True)

        # No emoji here on purpose: emoji glyphs render in their own fixed colors (mostly
        # white/light) regardless of CSS `color`, so against a light background they end up
        # looking washed out — text alone is more legible.
        render_header("MOODY TUNES", "Get song recommendations based on your face mood")
        st.divider()

        upload_tab, camera_tab = st.tabs([":material/upload_file: Upload a photo", ":material/photo_camera: Take a photo"])
        with upload_tab:
            uploaded_file = st.file_uploader("", type=["jpg", "png", "jpeg"], label_visibility="collapsed")
        with camera_tab:
            camera_file = st.camera_input("", label_visibility="collapsed")

        # Whichever widget the user actually used wins (only one will be non-None at a time
        # in normal use, but prefer a fresh camera shot if somehow both are set).
        active_file = camera_file if camera_file is not None else uploaded_file

        if active_file is not None:
            # Convert the uploaded/captured file to an OpenCV image
            # .getvalue() (not .read()): Streamlit reruns the whole script on every
            # interaction and reuses the same UploadedFile object across reruns — .read()
            # advances an internal cursor, so a second read (e.g. after a rerun triggered by
            # the tabs widget) returns empty bytes, which crashes cv2.imdecode with
            # "!buf.empty()" instead of decoding the image. getvalue() always returns the
            # full buffer regardless of prior reads.
            file_bytes = np.asarray(bytearray(active_file.getvalue()), dtype=np.uint8)
            cv_image = cv2.imdecode(file_bytes, 1)  # 1 indicates loading the image in color

            if cv_image is not None:
                # Only redo detection (and the Cloudinary upload / Spotify playlist below) when
                # this is a genuinely new photo. Interacting with a widget further down (like
                # the Shuffle button) also triggers a full script rerun, but active_file stays
                # the same object — file_id is stable across reruns of the same upload, so it's
                # the right key to detect "actually new photo" vs "just rerunning".
                if st.session_state.get("mt_file_id") != active_file.file_id:
                    with st.spinner("Detecting emotion from the photo..."):
                        detected_emotion, rgb_image, prediction = detect_emotion(cv_image)
                    st.session_state["mt_file_id"] = active_file.file_id
                    st.session_state["mt_detected_emotion"] = detected_emotion
                    st.session_state["mt_rgb_image"] = rgb_image
                    st.session_state["mt_prediction"] = prediction
                    st.session_state["mt_recommended_songs"] = None
                    st.session_state["mt_playlist_done"] = False

                detected_emotion = st.session_state["mt_detected_emotion"]
                rgb_image = st.session_state["mt_rgb_image"]
                prediction = st.session_state["mt_prediction"]

                if detected_emotion is not None:
                    styled_message(':material/check_circle: Great job!')

                    # Save the image on Cloudinary — once per photo, not on every shuffle rerun.
                    if not st.session_state["mt_playlist_done"]:
                        timestamp = time.strftime("%Y%m%d-%H%M%S")
                        picture_filename = f"{detected_emotion}---{timestamp}.jpg"
                        save_image_on_cloudinary(rgb_image, picture_filename)

                    with st.container(border=True):
                        # Display the uploaded and processed image
                        st.image(rgb_image, use_container_width=True)

                        # Confidence across all 7 emotions, not just the winning one — the
                        # model already computes this, it just wasn't being shown before.
                        if prediction is not None:
                            st.caption("Confidence by emotion")
                            confidence_df = pd.DataFrame(
                                {"Confidence (%)": prediction * 100}, index=emotion_labels
                            )
                            st.bar_chart(confidence_df, height=200, color="#2E1B16")

                        # Create a container for the recommended songs and subheader
                        st.subheader(f"For your {detected_emotion} mood, your tunes are:")
                        if st.session_state["mt_recommended_songs"] is None:
                            st.session_state["mt_recommended_songs"] = get_recommendations(detected_emotion, songs)
                        recommended_songs = st.session_state["mt_recommended_songs"]

                        if not recommended_songs.empty:
                            st.dataframe(recommended_songs[['Track', 'Artist']], use_container_width=True, hide_index=True)

                            if st.button(":material/shuffle: Shuffle songs"):
                                st.session_state["mt_recommended_songs"] = get_recommendations(detected_emotion, songs)
                                st.rerun()

                            # Playlist is created once per photo (not re-created on shuffle —
                            # shuffling only changes what's displayed here).
                            if not st.session_state["mt_playlist_done"]:
                                try:
                                    create_spotify_playlist(recommended_songs, SPOTIFY_USER_ID, detected_emotion)
                                except spotipy.SpotifyBaseException:
                                    styled_message("Couldn't auto-create a Spotify playlist right now (the Spotify connection needs to be re-authorized), but here are your song recommendations above! :material/music_note:")
                                st.session_state["mt_playlist_done"] = True
                else:
                    styled_message(':material/sentiment_dissatisfied: No face detected in the photo. Try again!')
            else:
                styled_message('Unable to read the photo. Please try again, folks!')

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
        styled_message("For more information, please reach out to diogoacapitao@gmail.com")
if __name__ == "__main__":
    main()