from docutils.nodes import system_message
from huggingface_hub import login
from transformers import AutoTokenizer, AutoModelForCausalLM, TextStreamer, BitsAndBytesConfig
import os
import torch

class MinuteGenerator:
    def __init__(self):
        self.token = os.getenv("HF_TOKEN")
        self.model = "meta-llama/Meta-Llama-3.1-8B-Instruct"
        login(token=self.token)

    def generate_minute(self, transcription: str) -> str:
        cache_dir = '/content/cache'

        messages = self.build_messages(transcription)
        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_quant_type="nf4"
        )

        tokenizer = AutoTokenizer.from_pretrained(self.model, cache_dir = cache_dir)
        tokenizer.pad_token = tokenizer.eos_token
        inputs = tokenizer.apply_chat_template(messages, return_tensors="pt").to("cuda")
        streamer = TextStreamer(tokenizer)
        model = AutoModelForCausalLM.from_pretrained(
            self.model,
            quantization_config=quant_config,
            device_map="auto",
            cache_dir = cache_dir
        )

        prompt_len = inputs.shape[1]
        outputs = model.generate(inputs, max_new_tokens=2000, streamer=streamer)

        generated_tokens = outputs[0][prompt_len:]

        return tokenizer.decode(generated_tokens, skip_special_tokens=True)


    def build_messages(self, transcription):
        system_message = "You are an assistant that produces minutes of meetings from transcripts, with summary, key discussion points, takeaways and action items with owners, in markdown."
        user_message = f"Below is an extract transcript of a Denver council meeting. Please write minutes in markdown, including a summary with attendees, location and date; discussion points; takeaways; and action items with owners.\n{transcription}"
        return [
            {"role": "system", "content": system_message},
            {"role": "user", "content": user_message}
        ]
