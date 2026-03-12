from typing import List
import os

import chainlit as cl
from groq import AsyncGroq

print("Initializing Groq client...")

api_key = os.getenv("GROQ_API_KEY")
if not api_key:
    raise RuntimeError(
        "GROQ_API_KEY environment variable is not set. "
        "Set it before running: export GROQ_API_KEY='gsk_...'"
    )

client = AsyncGroq(api_key=api_key)

print("Client initialized.")


def clean_response(text: str) -> str:
    """
    Removes any accidental role continuation like 'User:' from the model output.
    """
    if "User:" in text:
        text = text.split("User:")[0]
    return text.strip()


def get_prompt(instruction: str, history: List[str]) -> str:
    system = (
        "You are a friendly, helpful AI assistant for relatives and loved ones of military veterans.\n"
        "You answer general questions about veterans' mental and physical health, well-being, and daily life.\n"
        "You do NOT have access to any specific veteran's medical records or personal information.\n\n"
        "Guidelines:\n"
        "- Focus on supporting the relative: their concerns, feelings, and questions.\n"
        "- Give clear, practical suggestions on how to support and communicate with a veteran.\n"
        "- You may explain general concepts from health reports or medical/psychological terms in simple language.\n"
        "- Always stay general: do NOT interpret or diagnose a specific person's condition or test results.\n"
        "- Do NOT give medical advice, diagnoses, or treatment recommendations.\n"
        "- Encourage talking to qualified healthcare professionals for any specific medical or crisis concerns.\n\n"
        "Keep responses clear, empathetic, and reasonably short.\n"
    )

    prompt = f"### System:\n{system}\n\n"

    if history:
        for turn in history:
            prompt += f"{turn}\n"

    prompt += f"\nUser: {instruction}\nAssistant:"
    return prompt


@cl.on_chat_start
def on_chat_start():
    cl.user_session.set("message_history", [])


@cl.on_message
async def on_message(message: cl.Message):
    message_history = cl.user_session.get("message_history")

    msg = cl.Message(content="")
    await msg.send()

    prompt = get_prompt(message.content, message_history)

    response = ""

    stream = await client.chat.completions.create(
        model="llama-3.1-8b-instant",  # Groq model
        messages=[
            {"role": "user", "content": prompt},
        ],
        max_tokens=120,
        temperature=0.7,
        stream=True,
    )

    async for chunk in stream:
        delta = chunk.choices[0].delta.content or ""
        if not delta:
            continue
        await msg.stream_token(delta)
        response += delta

    await msg.update()

    clean = clean_response(response)

    message_history.append(f"User: {message.content}")
    message_history.append(f"Assistant: {clean}")

    MAX_TURNS = 4
    if len(message_history) > MAX_TURNS * 2:
        message_history = message_history[-MAX_TURNS * 2:]

    cl.user_session.set("message_history", message_history)
