"""Shared prompt construction so training and inference use the identical format."""

SYSTEM_PROMPT = (
    "You are a medical writing assistant. Rewrite the given biomedical sentence "
    "in plain, simple language that a patient with no medical background could "
    "understand. Keep the same meaning and do not add new information. Reply "
    "with only the rewritten sentence."
)


def build_chat_messages(source_sentence: str, target_sentence: str | None = None):
    """Return a list of chat messages. If target_sentence is given, an assistant
    turn is appended (for building training examples); otherwise the messages end
    after the user turn (for generation)."""
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"Sentence: {source_sentence}"},
    ]
    if target_sentence is not None:
        messages.append({"role": "assistant", "content": target_sentence})
    return messages
