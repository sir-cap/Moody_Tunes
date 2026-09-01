# MoodyTunes
    
    Personalized song recommendations based on your mood
    Data Analytics Bootcamp final project at Ironhack Portugal

    🔗 Live demo: https://moodytunes.streamlit.app
    (free-tier hosting — the app sleeps after inactivity, first load can take a minute to wake up)
    
    
## About
    
    MoodyTunes is a web app that harnesses the power of machine learning and facial 
    recognition technology to deliver tailored song recommendations based on your 
    current mood. By analyzing your facial expressions, MoodyTunes accurately identifies 
    your mood and curates songs that perfectly matches your emotional state.
    
    With MoodyTunes, discovering music that resonates with your mood has never been easier. 
    Whether you're feeling happy, sad, disgust, fear, angry,  surprise or 
    neutral, our intelligent machine learning model translates your facial cues into 
    distinct emotional categories. By leveraging this technology, we present you with an 
    extensive collection of songs carefully selected to uplift, inspire, and evoke the 
    emotions you're experiencing.
    
## Key Features:
    
    - Facial Recognition:
    A DNN face detector (OpenCV YuNet) finds your face, then a small CNN classifies your
    expression into one of 7 moods.
  
    - Upload or capture:
    Upload a photo, or take one directly with your camera (desktop or mobile browser).

    - Personalized Recommendations: 
    Based on your detected mood, MoodyTunes generates customized song 
    recommendations tailored specifically to you — with a Shuffle button to re-roll
    the list without re-detecting your mood.

    - Confidence breakdown:
    See how confident the model is across all 7 moods, not just the winning one.

    - Spotify Playlist: 
    Creates a playlist on Spotify based on the personalized recommendations
    
    Note: The facial recognition technology used by MoodyTunes respects your privacy and does 
    not share any personal data. Your facial expressions are 
    solely used for the purpose of determining your mood and 
    providing accurate song recommendations.
    
    Embrace the harmony between your mood and music with MoodyTunes today!

## 2026 update

The original 2023 bootcamp project stopped working (dependency drift — see commit history
for the full list of fixes) and got a round of improvements, done with AI pair-programming
via [Claude Code](https://claude.com/claude-code):

- Fixed the broken deployment (unpinned dependencies had drifted to incompatible versions:
  TensorFlow/Keras, OpenCV, a removed Streamlit API).
- Replaced the Haar Cascade face detector with OpenCV's YuNet DNN detector — more reliable
  detection.
- Added camera capture (`st.camera_input`) alongside file upload.
- Added a per-emotion confidence chart and a Shuffle-songs button.
- Found and removed three separate accidental credential leaks in the git history
  (`secrets.toml`, a stale OAuth token cache, and a compiled `.pyc` with hardcoded keys) —
  see [`SECRETS.md`](SECRETS.md) for how secrets are managed now.
- Rebuilt the visual design (color palette, responsive header, custom-styled alerts).

**Stack:** Python, Streamlit, TensorFlow/Keras, OpenCV, Spotify Web API (spotipy), Cloudinary,
deployed on Streamlit Community Cloud with GitHub auto-deploy.
    
## Data
    
    Songs: 
    https://www.kaggle.com/datasets/musicblogger/spotify-music-data-to-identify-the-moods
    
    Facial Recognition: 
    https://www.kaggle.com/datasets/jonathanoheix/face-expression-recognition-dataset
    
    
## Contact
    
    For any further question or comment you can contact me 
    through https://www.linkedin.com/in/diogocapitao/