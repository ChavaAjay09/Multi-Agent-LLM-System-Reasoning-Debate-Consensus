# input_handlers/audio_input.py

import os
import tempfile
import whisper
import numpy as np
import scipy.io.wavfile as wav
from google.colab import files
from google.colab.output import eval_js
from base64 import b64decode
from IPython.display import display, Javascript, Audio

# --- Helper function for JavaScript-based recording in Colab ---
# This code uses the browser's microphone to record and sends the data back to Python.
RECORD_JS = """
const sleep = time => new Promise(resolve => setTimeout(resolve, time))
const b2text = blob => new Promise(resolve => {
  const reader = new FileReader()
  reader.onloadend = e => resolve(e.target.result)
  reader.readAsDataURL(blob)
})
var record = time => new Promise(async resolve => {
  stream = await navigator.mediaDevices.getUserMedia({ audio: true })
  recorder = new MediaRecorder(stream)
  chunks = []
  recorder.ondataavailable = e => chunks.push(e.data)
  recorder.start()
  await sleep(time)
  recorder.onstop = async ()=>{
    blob = new Blob(chunks)
    text = await b2text(blob)
    resolve(text)
  }
  recorder.stop()
})
"""

class AudioHandler:
    """
    Handles audio input within a Google Colab environment by providing methods
    for uploading files and recording live audio from the browser.
    """
    def __init__(self, model_name="base"):
        """
        Initializes the Whisper model.
        Args:
            model_name (str): The name of the Whisper model to load (e.g., "tiny", "base", "small").
        """
        print("▶️ Initializing Whisper model... (This may download the model on first run)")
        try:
            self.model = whisper.load_model(model_name)
            print("✅ Whisper model loaded successfully.")
        except Exception as e:
            raise RuntimeError(f"❌ Failed to load Whisper model: {str(e)}")

    def transcribe_audio_file(self, file_path):
        """
        Transcribes a given audio file using the loaded Whisper model.
        """
        if not os.path.exists(file_path):
            print(f"❌ File not found at path: {file_path}")
            return None

        print(f"📁 Transcribing file: {os.path.basename(file_path)}...")
        try:
            result = self.model.transcribe(file_path)
            print("🧠 Transcription Complete.")
            return result["text"]
        except Exception as e:
            print(f"❌ Error during transcription: {e}")
            return None

    def upload_and_transcribe(self):
        """
        Provides a file upload button in Colab, saves the uploaded file temporarily,
        and transcribes it.
        """
        print("📤 Please upload an audio file (.wav, .mp3, .m4a, etc.).")
        uploaded = files.upload()

        if not uploaded:
            print("⚠️ No file uploaded.")
            return None

        # Get the name of the first uploaded file
        file_name = next(iter(uploaded))
        
        # Write the uploaded bytes to a temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(file_name)[1]) as temp_f:
            temp_f.write(uploaded[file_name])
            temp_file_path = temp_f.name
        
        # Transcribe the temporary file
        transcribed_text = self.transcribe_audio_file(temp_file_path)

        # Clean up the temporary file
        os.remove(temp_file_path)
        
        return transcribed_text

    def record_and_transcribe(self, duration_ms=5000):
        """
        Records audio from the user's microphone in the browser for a specified duration,
        saves it temporarily, and transcribes it.
        
        Args:
            duration_ms (int): The recording duration in milliseconds (e.g., 5000 for 5 seconds).
        """
        print(f"🎤 Please allow microphone access. Recording for {duration_ms / 1000} seconds...")
        
        try:
            # Execute JS to record audio
            display(Javascript(RECORD_JS))
            js_output = eval_js(f'record({duration_ms})')
            
            # The output is a base64 encoded string, decode it
            audio_data = b64decode(js_output.split(',')[1])
            
            # Save the decoded audio to a temporary file
            with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_f:
                temp_f.write(audio_data)
                temp_file_path = temp_f.name

            print("📼 Recording complete. Transcribing...")
            # Transcribe the temporary audio file
            transcribed_text = self.transcribe_audio_file(temp_file_path)

            # Clean up the temporary file
            os.remove(temp_file_path)
            
            return transcribed_text
            
        except Exception as e:
            print(f"❌ Failed to record or transcribe audio: {e}")
            return None

# --- Example of how to use this in a Colab notebook ---
if __name__ == '__main__':
    print("--- Testing AudioHandler in Colab ---")

    # You need to install these libraries in your notebook first:
    # !pip install openai-whisper
    # !apt-get -qq install -y ffmpeg

    # 1. Initialize the handler
    handler = AudioHandler()

    # 2. Test uploading a file
    print("\n--- Test 1: Uploading a file ---")
    text_from_upload = handler.upload_and_transcribe()
    if text_from_upload:
        print(f"\n🗣️ Text from uploaded file: {text_from_upload}")

    # 3. Test recording from microphone
    print("\n--- Test 2: Recording from microphone ---")
    text_from_record = handler.record_and_transcribe(duration_ms=5000)
    if text_from_record:
        print(f"\n🗣️ Text from recording: {text_from_record}")
