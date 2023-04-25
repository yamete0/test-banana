import whisper
import os
import subprocess
import base64
from io import BytesIO


# Init is ran on server startup
# Load your model to GPU as a global variable here using the variable name "model"
def init():
    global model
    model_name = os.getenv("MODEL_NAME")
    model = whisper.load_model(model_name, device="cuda", in_memory=True, fp16=True)


def downloadYTaudio(url, start_time, end_time, audio_file):
    # Download audio file
    audio_cmd = f'yt-dlp -f "bestaudio[ext=m4a]" --external-downloader ffmpeg --external-downloader-args "ffmpeg_i:-ss {start_time} -to {end_time}" -o "{audio_file}" "{url}"'
    audio_output = subprocess.run(audio_cmd, shell=True, capture_output=True, text=True)

    return (
        audio_output.returncode,
        audio_output.stderr,
    )


# Inference is ran for every server call
# Reference your preloaded global model variable here.
def inference(model_inputs: dict) -> dict:
    global model

    # Access URL from model_inputs dictionary
    url = model_inputs["url"]

    # Access start time from model_inputs dictionary
    start_time = model_inputs["start_time"]

    # Access end time from model_inputs dictionary
    end_time = model_inputs["end_time"]

    audio_file = "audio-youtube.m4a"

    audio_returncode, audio_stderr = downloadYTaudio(
        url, start_time, end_time, audio_file
    )

    if audio_returncode != 0:
        print(f"Error Audio Download: {audio_stderr}")
        return audio_stderr

    kwargs = {"beam_size": 5, "temperature": [0, 0.2, 0.4, 0.6, 0.8, 1]}

    # Run the model
    result = model.transcribe(audio_file, fp16=True, **kwargs)
    result["segments"] = [
        {
            "id": x["id"],
            "seek": x["seek"],
            "start": x["start"],
            "end": x["end"],
            "text": x["text"],
        }
        for x in result["segments"]
    ]
    os.remove(audio_file)
    # Return the results as a dictionary
    return result
