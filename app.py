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

# Inject JavaScript for webcam capture
st.markdown(
    """
    <script>
    async function captureImage() {
        const video = document.createElement('video');
        const canvas = document.createElement('canvas');
        const context = canvas.getContext('2d');

        const stream = await navigator.mediaDevices.getUserMedia({ video: true });
        video.srcObject = stream;
        await video.play();

        canvas.width = video.videoWidth;
        canvas.height = video.videoHeight;
        context.drawImage(video, 0, 0, canvas.width, canvas.height);

        const imageDataURL = canvas.toDataURL('image/png');
        stream.getTracks().forEach(track => track.stop());
        
        const imageInput = document.getElementById('imageData');
        imageInput.value = imageDataURL;
        document.getElementById('imageForm').submit();
    }
    </script>
    """,
    unsafe_allow_html=True,
)

# Hidden form to handle image submission
st.markdown(
    """
    <form id="imageForm" method="post">
        <input type="hidden" id="imageData" name="imageData">
    </form>
    <button onclick="captureImage()">Capture Image</button>
    """,
    unsafe_allow_html=True,
)

# Handle image submission
if st.experimental_get_query_params().get("imageData"):
    import base64
    from PIL import Image
    from io import BytesIO

    image_data = st.experimental_get_query_params()["imageData"][0]
    image_data = image_data.split(",")[1]
    image_bytes = base64.b64decode(image_data)
    image = Image.open(BytesIO(image_bytes))

    # Save the image for processing
    temp_image = NamedTemporaryFile(delete=False, suffix=".png")
    image.save(temp_image.name)

    # Call the image captioning API
    def query_image(image_file):
        headers = {"Authorization": f"Bearer {API_KEY}"}
        response = requests.post(API_URL, headers=headers, files={"file": image_file})
        return response.json()

    try:
        with st.spinner("Generating Caption..."):
            result = query_image(temp_image.name)
            if isinstance(result, list) and len(result) > 0:
                caption = result[0].get("generated_text", "No caption generated")
            else:
                caption = "Unexpected response from the model"

            # Display the image and caption
            st.image(image, caption=caption, use_column_width=True)
            st.success(f"Caption: {caption}")

            # Generate and autoplay audio
            def generate_audio(caption, lang):
                try:
                    tts = gTTS(text=caption, lang=lang)
                    temp_audio = NamedTemporaryFile(delete=False, suffix=".mp3")
                    tts.save(temp_audio.name)
                    return temp_audio.name
                except Exception as e:
                    st.error(f"Error in TTS: {e}")
                    return None

            audio_file_path = generate_audio(caption, language)
            if audio_file_path:
                audio_data_url = f"data:audio/mp3;base64,{base64.b64encode(open(audio_file_path, 'rb').read()).decode()}"
                st.markdown(
                    f"""
                    <audio autoplay>
                        <source src="{audio_data_url}" type="audio/mp3">
                        Your browser does not support the audio element.
                    </audio>
                    """,
                    unsafe_allow_html=True,
                )
    except Exception as e:
        st.error(f"Error processing image: {e}")
