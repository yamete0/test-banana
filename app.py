import torch
import whisper
import os
import base64
from io import BytesIO

# Init is ran on server startup
# Load your model to GPU as a global variable here using the variable name "model"
def init():
    global model
    model_name = os.getenv("MODEL_NAME")
    model = whisper.load_model(model_name, device="cuda", in_memory=True)

def _parse_arg(args : str, data : dict, default : None):
    arg = data.get(args, None)
    if arg == None:
        if default is None:
            raise Exception(f"Missing required argument: {args}")
        else:
            return default

# Inference is ran for every server call
# Reference your preloaded global model variable here.
def inference(model_inputs:dict) -> dict:
    global model

    # Parse out your arguments
    try:
        mp3BytesString = _parse_arg("mp3BytesString", model_inputs)
        beam_size = _parse_arg("beam_size", model_inputs, 5)
        fp16 = _parse_arg("fp16", model_inputs, True)

    except Exception as e:
        return {"error":str(e)}
    
    mp3Bytes = BytesIO(base64.b64decode(mp3BytesString.encode("ISO-8859-1")))
    with open('input.mp3','wb') as file:
        file.write(mp3Bytes.getbuffer())
    
    # Run the model
    result = model.transcribe("input.mp3", fp16=fp16, beam_size=beam_size)
    os.remove("input.mp3")
    # Return the results as a dictionary
    return result
