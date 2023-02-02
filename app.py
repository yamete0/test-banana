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

def _parse_arg(args : str, data : dict, default = None):
    arg = data.get(args, None)
    if arg == None:
        if default is None:
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
        opusBytesString = _parse_arg("opusBytesString", model_inputs)
        beam_size = _parse_arg("beam_size", model_inputs, 5)
        fp16 = _parse_arg("fp16", model_inputs, True)

    except Exception as e:
        return {"error":str(e)}

    print(f"Settings: beam_size={beam_size}, fp16={fp16}")
    
    opusBytes = BytesIO(base64.b64decode(opusBytesString.encode("ISO-8859-1")))
    tmp_file = "input.opus"
    with open(tmp_file,'wb') as file:
        file.write(opusBytes.getbuffer())
    
    # Run the model
    result = model.transcribe(tmp_file, fp16=fp16, beam_size=beam_size)
    os.remove(tmp_file)
    # Return the results as a dictionary
    return result
