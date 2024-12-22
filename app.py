import streamlit as st
from gtts import gTTS
import requests
from tempfile import NamedTemporaryFile

# Constants
API_URL = "https://api-inference.huggingface.co/models/Salesforce/blip-image-captioning-large"
API_KEY = "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"

# Streamlit app title
st.title("Image Captioning for the Visually Impaired")

# Language selection
LANGUAGES = {
    "en": "English",
    "ta": "Tamil",
    "hi": "Hindi",
    "te": "Telugu",
    "ml": "Malayalam",
}
language = st.selectbox("Select Language", options=list(LANGUAGES.keys()), format_func=lambda x: LANGUAGES[x])

# Upload image
uploaded_file = st.file_uploader("Capture or Upload an Image", type=["jpg", "jpeg", "png"])

def query_image(image_file):
    headers = {"Authorization": f"Bearer {API_KEY}"}
    response = requests.post(API_URL, headers=headers, files={"file": image_file})
    return response.json()

def generate_audio(caption, lang):
    try:
        tts = gTTS(text=caption, lang=lang)
        temp_file = NamedTemporaryFile(delete=False, suffix='.mp3')
        tts.save(temp_file.name)
        return temp_file.name
    except Exception as e:
        st.error(f"Error in TTS: {e}")
        return None

# Process uploaded file
if uploaded_file is not None:
    with st.spinner("Generating Caption..."):
        try:
            # Call the captioning API
            result = query_image(uploaded_file)
            if isinstance(result, list) and len(result) > 0:
                caption = result[0].get("generated_text", "No caption generated")
            else:
                caption = "Unexpected response from the model"

            # Display the caption
            st.image(uploaded_file, caption=caption, use_column_width=True)
            st.success(f"Caption: {caption}")

            # Generate and display audio
            audio_file_path = generate_audio(caption, language)
            if audio_file_path:
                with open(audio_file_path, "rb") as audio_file:
                    audio_data = audio_file.read()
                    st.audio(audio_data, format="audio/mp3")
        except Exception as e:
            st.error(f"Error processing image: {e}")
