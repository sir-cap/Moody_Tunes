import pickle
from pathlib import Path
import streamlit as st
import streamlit_authenticator as stauth

names = ['Diogo', 'Luke', 'Joana']
usernames =['Cap','Luke','Jo'] 
passwords=['cap123', 'luke123', 'jo123']

hashed_passwords = stauth.Hasher(passwords).generate() # bycrat algor
