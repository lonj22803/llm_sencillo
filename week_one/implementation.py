import gradio as gr
import transformers
from transformers import logging
logging.set_verbosity_error()
import torch
print("CUDA available: ", torch.cuda.is_available())
print("Number of GPUs: ", torch.cuda.device_count())
model_id = "meta-llama/Meta-Llama-3.1-8B-Instruct"
pipeline = transformers.pipeline(
    "text-generation",
    model=model_id,
    device_map="auto"
)
chat_history = [
{
  "role": "system",
  "content": "You are a tourism assistant who always responds in Spanish. You are a specialist in the Madrid Metro."
}

    ]
def format_history(history):
    conversation = ""
    for msg in history:
        if msg["role"] == "system":
            conversation += f"Sistema: {msg['content']}\n"
        elif msg["role"] == "user":
            conversation += f"Usuario: {msg['content']}\n"
        elif msg["role"] == "assistant":
            conversation += f"Asistente: {msg['content']}\n"
    return conversation
def chat(user_input):
    chat_history.append({"role": "user", "content": user_input})
    input_text = format_history(chat_history) + "Asistente:"
    outputs = pipeline(
        input_text,
        max_new_tokens=256,
        do_sample=True,
        temperature=0.7,
        pad_token_id=50256
    )
    generated_text = outputs[0]["generated_text"]
    response = generated_text.split("Asistente:")[-1].strip()
    chat_history.append({"role": "assistant", "content": response})
    return response
with gr.Blocks() as demo:
    gr.Markdown("# Chatbot de Turismo en Español")
    chatbot = gr.Chatbot()
    msg = gr.Textbox(label="Escribe tu mensaje")
    clear = gr.Button("Limpiar chat")
    def respond(message, chat_list):
        response = chat(message)
        chat_list.append((message, response))
        return chat_list, ""
    msg.submit(respond, [msg, chatbot], [chatbot, msg])
    clear.click(lambda: ([], []), None, chatbot)
demo.launch()