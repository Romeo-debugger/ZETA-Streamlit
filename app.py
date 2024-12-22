import streamlit as st
from streamlit_webrtc import webrtc_streamer
import av
from PIL import Image
import numpy as np
from gtts import gTTS
import os
from transformers import BlipProcessor, BlipForConditionalGeneration
from translate import Translator
import tempfile
import pygame.mixer
import time

# Initialize audio mixer
pygame.mixer.init()

# Initialize the BLIP model and processor
@st.cache_resource
def load_model():
    processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
    model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
    return processor, model

processor, model = load_model()

# Language configuration (keeping the extensive language support from original code)
LANGUAGES = {
    # Indian Languages
    'en': {'name': 'English', 'code': 'en'},
    'ta': {'name': 'Tamil', 'code': 'ta'},
    'kn': {'name': 'Kannada', 'code': 'kn'},
    'hi': {'name': 'Hindi', 'code': 'hi'},
    'ml': {'name': 'Malayalam', 'code': 'ml'},
    'te': {'name': 'Telugu', 'code': 'te'},
    # Adding more languages as needed...
}

# Initialize session state
if 'current_language' not in st.session_state:
    st.session_state['current_language'] = 'en'
if 'caption' not in st.session_state:
    st.session_state['caption'] = ""

def translate_text(text, target_lang):
    """Translate text to target language"""
    try:
        if target_lang == 'en':
            return text
        translator = Translator(to_lang=target_lang)
        translation = translator.translate(text)
        return translation
    except Exception as e:
        st.error(f"Translation error: {e}")
        return text

def speak_caption(caption, language):
    """Generate and play audio for the caption"""
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp3') as temp_file:
            tts = gTTS(text=caption, lang=language)
            tts.save(temp_file.name)
            st.audio(temp_file.name)
            # Clean up
            os.unlink(temp_file.name)
    except Exception as e:
        st.error(f"Text-to-speech error: {e}")

def generate_caption(image: Image.Image) -> str:
    """Generate caption for the image"""
    try:
        inputs = processor(images=image, return_tensors="pt")
        outputs = model.generate(**inputs)
        caption = processor.decode(outputs[0], skip_special_tokens=True)
        return caption
    except Exception as e:
        st.error(f"Caption generation error: {e}")
        return "Error generating caption"

def video_frame_callback(frame):
    """Process video frames and generate captions"""
    img = frame.to_image()  # Convert frame to PIL Image
    
    # Generate caption
    caption = generate_caption(img)
    
    # Translate caption if needed
    if st.session_state['current_language'] != 'en':
        caption = translate_text(caption, st.session_state['current_language'])
    
    # Update session state
    st.session_state['caption'] = caption
    
    # Generate and play audio
    speak_caption(caption, LANGUAGES[st.session_state['current_language']]['code'])
    
    return av.VideoFrame.from_ndarray(np.array(img), format="rgb24")

# Streamlit UI
def main():
    st.title("Multilingual Image Captioning for Blind Users")
    st.markdown("This app captures images from your webcam, generates captions, and plays them in multiple languages.")
    
    # Language selector
    selected_language = st.sidebar.selectbox(
        "Select Language",
        options=list(LANGUAGES.keys()),
        format_func=lambda x: LANGUAGES[x]['name'],
        index=list(LANGUAGES.keys()).index(st.session_state['current_language'])
    )
    
    if selected_language != st.session_state['current_language']:
        st.session_state['current_language'] = selected_language
    
    # WebRTC streamer
    webrtc_streamer(
        key="caption_stream",
        video_frame_callback=video_frame_callback,
        media_stream_constraints={
            "video": True,
            "audio": False,
        },
        async_processing=True
    )
    
    # Display the latest caption
    st.text_area(
        "Generated Caption",
        value=st.session_state['caption'],
        height=100,
        key="caption_display"
    )
    
    # Additional settings and information
    with st.sidebar:
        st.markdown("### Settings")
        st.checkbox("Auto-refresh captions", value=True)
        st.markdown("### About")
        st.markdown("""
        This application combines real-time image captioning with:
        - Multi-language support
        - Text-to-speech functionality
        - Real-time webcam processing
        - Automatic caption generation
        """)

if __name__ == "__main__":
    main()
