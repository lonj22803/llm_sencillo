import asyncio
import gradio as gr
from ollama import AsyncClient

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

async def generate_response_async():
    # Construir mensajes para enviar a ollama
    messages = chat_history.copy()
    # Usamos AsyncClient para obtener la respuesta en streaming
    client = AsyncClient()
    response_content = ""
    async for part in await client.chat(model="llama3", messages=messages, stream=True):
        response_content += part["message"]["content"]
    return response_content

def chat(user_input):
    # Añadir mensaje del usuario al historial
    chat_history.append({"role": "user", "content": user_input})
    # Ejecutar la generación de respuesta asincrónica
    response = asyncio.run(generate_response_async())
    # Añadir respuesta del asistente al historial
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
