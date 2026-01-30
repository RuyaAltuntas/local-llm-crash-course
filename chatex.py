from typing import List
from ctransformers import AutoModelForCausalLM
import chainlit as cl


print("Loading LLM...")

llm = AutoModelForCausalLM.from_pretrained(
    "zoltanctoth/orca_mini_3B-GGUF",
    model_file="orca-mini-3b.q4_0.gguf"
)

print("LLM loaded successfully.")


def clean_response(text: str) -> str:
    """
    Removes any accidental role continuation like 'User:' from the model output.
    """
    if "User:" in text:
        text = text.split("User:")[0]
    return text.strip()


def get_prompt(instruction: str, history: List[str]) -> str:
    system = (
        "You are a friendly helpful AI assistant designed for veterans.\n"
        "You chat about daily life, military memories, and personal experiences.\n"
        "You help users cope with stress, loneliness, and low mood in a natural, human way.\n\n"

        "Guidelines:\n"
        "- Be kind, supportive, and conversational.\n"
        "- It is okay to give advice and ask thoughtful follow-up questions only if needed.\n"
        "- You may sound reflective, like a good listener.\n"
        "- Do NOT give medical advice, diagnoses, or treatment recommendations.\n\n"

        "Keep responses clear and reasonably short.\n"
    )

    prompt = f"### System:\n{system}\n\n"

    # ✅ Role-separated memory
    if history:
        for turn in history:
            prompt += f"{turn}\n"

    # 🔑 IMPORTANT: model must stop after Assistant response
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
    for token in llm(
        prompt,
        stream=True,
        max_new_tokens=120,
        temperature=0.7,
        stop=["User:"]  # stop self-dialogue
    ):
        await msg.stream_token(token)
        response += token

    await msg.update()

    # Clean assistant output before saving
    clean = clean_response(response)

    message_history.append(f"User: {message.content}")
    message_history.append(f"Assistant: {clean}")

    # Keep last N turns only
    MAX_TURNS = 4
    if len(message_history) > MAX_TURNS * 2:
        message_history = message_history[-MAX_TURNS * 2:]

    cl.user_session.set("message_history", message_history)
