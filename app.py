import streamlit as st
from streamlit_webrtc import webrtc_streamer
import av
from PIL import Image
import numpy as np
from gtts import gTTS
import os
from transformers import BlipProcessor, BlipForConditionalGeneration

# Load the BLIP model and processor
processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")

# Function to generate captions
def generate_caption(image: Image.Image) -> str:
    inputs = processor(images=image, return_tensors="pt")
    outputs = model.generate(**inputs)
    caption = processor.decode(outputs[0], skip_special_tokens=True)
    return caption

# Function to process video frames
def video_frame_callback(frame):
    img = frame.to_image()  # Convert frame to PIL Image
    caption = generate_caption(img)

    # Streamlit to display the caption
    st.session_state['caption'] = caption

    # Generate speech using gTTS and save it as an audio file
    tts = gTTS(caption)
    tts.save("caption.mp3")

    # Play the audio
    st.audio("caption.mp3", format="audio/mp3", start_time=0)
    return av.VideoFrame.from_ndarray(np.array(img), format="rgb24")

# Streamlit app setup
st.title("Image Captioning for Blind Users")
st.markdown("This app captures images from your webcam, generates captions, and plays them aloud.")

# We use the session state to store the last caption
if 'caption' not in st.session_state:
    st.session_state['caption'] = ""

webrtc_streamer(
    key="caption_stream",
    video_frame_callback=video_frame_callback,
    media_stream_constraints={
        "video": True,
        "audio": False,
    },
    async_processing=False,
)

# Display the latest caption
st.text_area("Generated Caption", value=st.session_state['caption'], height=100)
