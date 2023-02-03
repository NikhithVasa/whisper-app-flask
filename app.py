from flask import Flask, abort, request
from tempfile import NamedTemporaryFile
import base64
import whisper
import torch

# Check if NVIDIA GPU is available
torch.cuda.is_available()
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Load the Whisper model:
model = whisper.load_model("base", device=DEVICE)

app = Flask(__name__)


@app.route("/")
def hello():
    return "Whisper Hello World!"


@app.route('/whisper', methods=['POST'])
def handler():
    if not request.json or not request.json.get('audio_data'):
        # If the user didn't submit audio data, return a 400 (Bad Request) error.
        abort(400)

    # Get the base64 encoded audio data
    audio_data = request.json.get('audio_data')

    # Create a temporary file.
    # The location of the temporary file is available in `temp.name`.
    temp = NamedTemporaryFile()

    # Decode the base64 encoded audio data and write it to the temporary file.
    temp.write(base64.b64decode(audio_data))
    temp.seek(0)

    # Let's get the transcript of the temporary file.
    result = model.transcribe(temp.name)

    # This will be automatically converted to JSON.
    return {'transcript': result['text']}
