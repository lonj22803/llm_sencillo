import transformers
from transformers import logging
logging.set_verbosity_error()  # Suppress warnings and info messages
import torch

print("CUDA available: ", torch.cuda.is_available())
print("Number of GPUs: ", torch.cuda.device_count())
model_id = "meta-llama/Meta-Llama-3.1-8B-Instruct"

pipeline = transformers.pipeline(
    "text-generation",
    model=model_id,
    device_map="auto"

)

messages = [
    {"role": "system", "content": "You are a chatbot who always responds in spanish and a tourism assistant."},
    {"role": "user", "content": "Hola, quiero ir a Madrid pero no se a donde ir"},
]

outputs = pipeline(
    messages,
    max_new_tokens=256,
)
print(outputs[0]["generated_text"][-1])
