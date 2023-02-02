import whisper
import os
import base64
from io import BytesIO

# Init is ran on server startup
# Load your model to GPU as a global variable here using the variable name "model"
def init():
    global model
    model_name = os.getenv("MODEL_NAME")
    model = whisper.load_model(model_name, device="cuda", in_memory=True, fp16=True)

def _parse_arg(args : str, data : dict, default = None, required = False):
    arg = data.get(args, None)
    if arg == None:
        if required:
            raise Exception(f"Missing required argument: {args}")
        else:
            return default

    return arg

# Inference is ran for every server call
# Reference your preloaded global model variable here.
def inference(model_inputs:dict) -> dict:
    global model

    # Parse out your arguments
    try:
        BytesString = _parse_arg("base64String", model_inputs, required=True)
        beam_size = _parse_arg("beam_size", model_inputs, None)
        best_of = _parse_arg("best_of", model_inputs, 5)
        audio_type = _parse_arg("audio_type", model_inputs, "opus")
        temperature = _parse_arg("temperature", model_inputs, (0.0, 0.2, 0.7))
        initial_prompt = _parse_arg("initial_prompt", model_inputs, None)
        if audio_type not in ["opus", "wav", "flac", "mp3", "m4a"]:
            raise Exception(f"Invalid audio_type: {audio_type}")

    except Exception as e:
        return {"error":str(e)}

    print(f"Settings: beam_size={beam_size}")
    
    bytes = BytesIO(base64.b64decode(BytesString.encode("ISO-8859-1")))

    tmp_file = "input."+audio_type
    with open(tmp_file,'wb') as file:
        file.write(bytes.getbuffer())
    
    # Run the model
    result = model.transcribe(tmp_file, fp16=True,
        temperature=temperature,
        beam_size=beam_size,
        best_of=best_of,
        initial_prompt=initial_prompt,
        )
    os.remove(tmp_file)
    # Return the results as a dictionary
    return result
