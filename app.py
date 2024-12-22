import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase
import requests
import av
import time
from gtts import gTTS
import os
from translate import Translator
from io import BytesIO
import base64
from PIL import Image
import numpy as np

# Define constants
API_URL = "https://api-inference.huggingface.co/models/Salesforce/blip-image-captioning-large"
API_KEY = "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"

# Language configuration (same as original)
LANGUAGES = {
    # Indian Languages
    'en': {'name': 'English', 'code': 'en'},
    'ta': {'name': 'Tamil', 'code': 'ta'},
    'kn': {'name': 'Kannada', 'code': 'kn'},
    'hi': {'name': 'Hindi', 'code': 'hi'},
    'ml': {'name': 'Malayalam', 'code': 'ml'},
    'te': {'name': 'Telugu', 'code': 'te'},
    # Adding more languages would follow the same pattern
}

class VideoTransformer(VideoTransformerBase):
    def __init__(self):
        self.frame_count = 0
        self.last_capture_time = 0
        self.capture_interval = 3  # Capture every 3 seconds

    def transform(self, frame):
        current_time = time.time()
        img = frame.to_ndarray(format="bgr24")
        
        # Capture frame every 3 seconds
        if current_time - self.last_capture_time >= self.capture_interval:
            self.last_capture_time = current_time
            # Convert BGR to RGB
            rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(rgb_img)
            
            # Save to buffer
            buffer = BytesIO()
            pil_img.save(buffer, format="JPEG")
            st.session_state['current_image'] = buffer.getvalue()
            
            # Generate caption
            if 'current_image' in st.session_state:
                generate_caption()

        return av.VideoFrame.from_ndarray(img, format="bgr24")

def translate_text(text, target_lang):
    try:
        if target_lang == 'en':
            return text
        translator = Translator(to_lang=target_lang)
        translation = translator.translate(text)
        return translation
    except Exception as e:
        st.error(f"Translation error: {str(e)}")
        return text

def query_huggingface(image_data):
    headers = {"Authorization": f"Bearer {API_KEY}"}
    response = requests.post(API_URL, headers=headers, data=image_data)
    return response.json()

def generate_caption():
    try:
        image_data = st.session_state['current_image']
        response = query_huggingface(image_data)
        
        if isinstance(response, list) and len(response) > 0:
            caption = response[0].get("generated_text", "No caption generated")
            translated_caption = translate_text(caption, st.session_state['current_language'])
            st.session_state['current_caption'] = translated_caption
            
            # Generate audio for the caption
            tts = gTTS(text=translated_caption, lang=st.session_state['current_language'])
            audio_fp = BytesIO()
            tts.write_to_fp(audio_fp)
            audio_bytes = audio_fp.getvalue()
            st.session_state['current_audio'] = audio_bytes
            
    except Exception as e:
        st.error(f"Error generating caption: {str(e)}")

def main():
    st.set_page_config(page_title="Project ZETA", layout="wide")
    st.title("Project ZETA")

    # Initialize session state
    if 'current_language' not in st.session_state:
        st.session_state['current_language'] = 'en'
    if 'current_caption' not in st.session_state:
        st.session_state['current_caption'] = ''
    if 'current_audio' not in st.session_state:
        st.session_state['current_audio'] = None

    # Sidebar for language selection
    with st.sidebar:
        st.header("Settings")
        selected_language = st.selectbox(
            "Select Language",
            options=list(LANGUAGES.keys()),
            format_func=lambda x: LANGUAGES[x]['name'],
            index=list(LANGUAGES.keys()).index(st.session_state['current_language'])
        )
        
        if selected_language != st.session_state['current_language']:
            st.session_state['current_language'] = selected_language
            if 'current_caption' in st.session_state and st.session_state['current_caption']:
                generate_caption()  # Regenerate caption in new language

    # Main content
    col1, col2 = st.columns([2, 1])

    with col1:
        st.header("Camera Feed")
        webrtc_ctx = webrtc_streamer(
            key="camera",
            video_transformer_factory=VideoTransformer,
            rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
            media_stream_constraints={"video": True, "audio": False}
        )

    with col2:
        st.header("Caption")
        if 'current_caption' in st.session_state and st.session_state['current_caption']:
            st.write(st.session_state['current_caption'])
            
            if 'current_audio' in st.session_state and st.session_state['current_audio']:
                st.audio(st.session_state['current_audio'], format='audio/mp3')

        if 'current_image' in st.session_state:
            st.image(st.session_state['current_image'], caption="Captured Image", use_column_width=True)

if __name__ == "__main__":
    main()
