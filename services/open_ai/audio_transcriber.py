from openai import OpenAI
import os

class AudioTranscriber:
    def __init__(self):
        self.api_key = os.getenv("OPENAI_API_KEY")
        self.model = 'whisper-1'
        self.client = OpenAI()

    def transcribe(self, audio_filename):
        audio_file = open(audio_filename, "rb")
        transcription = self.client.audio.transcriptions.create(model=self.model, file=audio_file, response_format="text")
        return transcription