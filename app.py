import whisper
import os
import subprocess
import base64
from io import BytesIO
import datetime
import random


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


def downloadYTClip(url, start_time, end_time, file_name):
    cmd = f'yt-dlp -f "bestvideo[height<=1080][ext=mp4]+bestaudio[ext=m4a]/best[height<=1080][ext=mp4]" --external-downloader ffmpeg --external-downloader-args "ffmpeg_i:-ss {start_time} -to {end_time}" -o "{file_name}" "{url}"'
    output = subprocess.run(cmd, shell=True, capture_output=True, text=True)

    return (
        output.returncode,
        output.stderr,
    )


def encodeFFMPEG(
    file_name, bottom_file, sound_file, sub_filename, clip_length, output_name
):
    cmd = f'ffmpeg -i {file_name} -i "{bottom_file}" -i "{sound_file}" -filter_complex "[0:v]scale=-2:880, crop=1080:880:420:0[v0];[1:v]crop=1080:1040:420:0[v1];[v0][v1]vstack=inputs=2,ass=\'{sub_filename}\'[v];[2:a]volume=0.2[a1];[0:a]volume=1.5[a2];[a2][a1]amix=inputs=2[a]" -map [v] -map [a] -c:v libx264 -c:a aac -t {clip_length} -y "{output_name}"'
    video_output = subprocess.run(cmd, shell=True, capture_output=True, text=True)

    return (video_output.returncode, video_output.stderr)


def transcribe_whisper(url, start_time, end_time):
    global model
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


def exportVid(url, start_time, end_time, subtitles):
    file_name = "input.mp4"

    returncode, stderr = downloadYTClip(url, start_time, end_time, file_name)
    if returncode != 0:
        error = f"Error Video Download: {stderr}"
        print(error)
        return error

    start_delta = datetime.timedelta(
        hours=int(start_time.split(":")[0]),
        minutes=int(start_time.split(":")[1]),
        seconds=int(start_time.split(":")[2]),
    )
    end_delta = datetime.timedelta(
        hours=int(end_time.split(":")[0]),
        minutes=int(end_time.split(":")[1]),
        seconds=int(end_time.split(":")[2]),
    )

    clip_length = (
        end_delta - start_delta - datetime.timedelta(seconds=1.5)
    ).total_seconds()

    bottom_file = f"bottom_clips/" + random.choice(
        os.listdir("bottom_clips")
    )  # get random video from bottom clips folder
    sound_file = f"audio/" + random.choice(
        os.listdir("audio")
    )  # get random music from audio folder

    sub_filename = "subtitles.ass"

    with open(sub_filename, "r") as sub_file:
        sub_file.writelines(subtitles)

    output_name = "encodedVideoFFMPEG.mp4"

    video_returncode, video_stderr, output_file = encodeFFMPEG(
        file_name, bottom_file, sound_file, sub_filename, clip_length, output_name
    )
    if video_returncode != 0:
        error = f"Error Encode Video: {video_stderr}"
        print(error)
        return error

    with open(output_file, "rb") as video_file:
        video_base64 = base64.b64encode(video_file.read()).decode("utf-8")
    # Remove the files after encoding
    os.remove(output_file)
    os.remove(sub_filename)

    return video_base64


# Inference is ran for every server call
# Reference your preloaded global model variable here.
def inference(model_inputs: dict) -> dict:
    # Access URL from model_inputs dictionary
    url = model_inputs["url"]

    # Access start time from model_inputs dictionary
    start_time = model_inputs["start_time"]

    # Access end time from model_inputs dictionary
    end_time = model_inputs["end_time"]

    type = model_inputs["type"]

    if type == "export_video":
        subtitles = model_inputs["subtitles_raw"]
        output = exportVid(url, start_time, end_time, subtitles)
        return output
    elif type == "transcribe_audio":
        output = transcribe_whisper(url, start_time, end_time)
        return output
    else:
        return "invalid type"
