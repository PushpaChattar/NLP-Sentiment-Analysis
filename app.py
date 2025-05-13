import streamlit as st
import pandas as pd
import os 
import joblib
from pathlib import Path


# Set the page title
st.set_page_config(page_title="Sentiment Anslysis App", page_icon="🖥️")

# App title and description

st.title("Sentiment Analysis App")
st.write("This app predicts the sentiment of text based on different sources.")
