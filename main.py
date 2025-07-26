from dotenv import load_dotenv
from panel.pane import Markdown
import gradio as gr

from services.open_ai.audio_transcriber import AudioTranscriber
from services.hugging_face.minute_generator import MinuteGenerator

load_dotenv()

def generate_minute(audio_file):
    transcriber = AudioTranscriber()
    minute_generator = MinuteGenerator()
    minute = minute_generator.generate_minute(transcriber.transcribe(audio_file))

    return minute

with gr.Blocks() as demo:
    file_output = gr.File()
    upload_button = gr.UploadButton("Sube un archivo de audio", file_types=["audio"])
    generate_button = gr.Button(value="Generar minuta")
    minute_placeholder = gr.Markdown(label="Minuta generada")

    generate_button.click(generate_minute, inputs=file_output, outputs=minute_placeholder)


demo.launch(share=True)