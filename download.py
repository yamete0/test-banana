# In this file, we define download_model
# It runs during container build time to get model weights built into the container

# In this example: A Huggingface BERT model

import whisper
from supabase import create_client, Client
import torch
import os


def download_model():
    model_name = os.getenv("MODEL_NAME")
    model = whisper.load_model(model_name)
    url: str = "https://nginpaisdlnwgdspewrq.supabase.co"
    key: str = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6Im5naW5wYWlzZGxud2dkc3Bld3JxIiwicm9sZSI6ImFub24iLCJpYXQiOjE2ODI4NzAxOTQsImV4cCI6MTk5ODQ0NjE5NH0.ahz7Y2PxLIdoWJSxC6iQip9NIVZaL04dwn4OcJTRfno"
    supabase: Client = create_client(url, key)


if __name__ == "__main__":
    download_model()
