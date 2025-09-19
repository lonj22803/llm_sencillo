import gradio as gr
import transformers
from transformers import logging
import torch

logging.set_verbosity_error()

print("CUDA available: ", torch.cuda.is_available())
print("Number of GPUs: ", torch.cuda.device_count())

model_id = "meta-llama/Meta-Llama-3.1-8B-Instruct"

pipeline = transformers.pipeline(
    "text-generation",
    model=model_id,
    device_map="auto"
)

# Parámetros para control de contexto
MAX_TOKENS = 1500
ULTIMAS_INTERACCIONES = 6  # Número de últimas interacciones completas que mantenemos

chat_history = [
    {
        "role": "system",
        "content": "You are a tourism assistant who always responds in Spanish. You are a specialist in the Madrid Metro."
    }
]

def contar_tokens(texto):
    # Aproximación simple: contar palabras
    return len(texto.split())

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

def generar_resumen(texto):
    """
    Genera un resumen del texto usando el mismo pipeline.
    Se limita la generación a 150 tokens para el resumen.
    """
    prompt = (
        "Resume brevemente la siguiente conversación, manteniendo los puntos clave:\n\n"
        f"{texto}\n\nResumen:"
    )
    outputs = pipeline(
        prompt,
        max_new_tokens=150,
        do_sample=False,
        temperature=0.3,
        pad_token_id=50256
    )
    resumen_generado = outputs[0]["generated_text"]
    # Extraer solo la parte del resumen (después de "Resumen:")
    resumen = resumen_generado.split("Resumen:")[-1].strip()
    return resumen

def resumir_contexto(history):
    """
    Resume las primeras partes del contexto para mantenerlo dentro del límite.
    """
    # Formatear solo la parte a resumir
    texto_a_resumir = format_history(history)
    resumen = generar_resumen(texto_a_resumir)
    # Crear un mensaje system con el resumen
    resumen_msg = {
        "role": "system",
        "content": f"Resumen de la conversación previa: {resumen}"
    }
    return resumen_msg

def controlar_contexto():
    """
    Controla la longitud del contexto, resumiendo si es necesario.
    """
    global chat_history
    # Formatear toda la conversación para contar tokens
    texto_completo = format_history(chat_history)
    if contar_tokens(texto_completo) > MAX_TOKENS:
        # Resumir todo menos las últimas N interacciones (cada interacción = user + assistant)
        # Calculamos cuántos mensajes son las últimas N interacciones
        mensajes_ultimas = ULTIMAS_INTERACCIONES * 2  # user + assistant por interacción
        # Excluir el primer mensaje system para resumir solo mensajes de usuario y asistente
        system_msg = chat_history[0]
        mensajes_a_resumir = chat_history[1:-mensajes_ultimas] if mensajes_ultimas < len(chat_history) else chat_history[1:]
        resumen_msg = resumir_contexto(mensajes_a_resumir)
        # Reconstruir el chat_history: system + resumen + últimas interacciones
        chat_history = [system_msg, resumen_msg] + chat_history[-mensajes_ultimas:]

def chat(user_input):
    global chat_history
    chat_history.append({"role": "user", "content": user_input})
    controlar_contexto()
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
