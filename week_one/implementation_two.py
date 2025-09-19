import gradio as gr
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

# Carga modelo y tokenizer
model_id = "meta-llama/Llama-3.1-8B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id).to("cuda" if torch.cuda.is_available() else "cpu")

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

    # Tokenizar entrada
    inputs = tokenizer(input_text, return_tensors="pt").to(model.device)

    # Generar respuesta
    outputs = model.generate(
        **inputs,
        max_new_tokens=256,
        do_sample=True,
        temperature=0.7,
        pad_token_id=tokenizer.eos_token_id
    )

    # Decodificar texto generado
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

    # Extraer solo la respuesta del asistente (después de "Asistente:")
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
