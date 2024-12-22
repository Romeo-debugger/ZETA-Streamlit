import streamlit as st
from streamlit_webrtc import webrtc_streamer
import av
from PIL import Image
import numpy as np
import pyttsx3
from transformers import BlipProcessor, BlipForConditionalGeneration

# Load the BLIP model and processor
processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")

# Initialize text-to-speech engine
engine = pyttsx3.init()
engine.setProperty('rate', 150)  # Adjust speaking speed

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

    # Read the caption aloud
    engine.say(caption)
    engine.runAndWait()
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
