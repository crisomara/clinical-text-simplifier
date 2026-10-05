"""Tests for the shared prompt builder used by training, evaluation and the app."""

from prompting import SYSTEM_PROMPT, build_chat_messages


def test_generation_prompt_ends_after_user_turn():
    messages = build_chat_messages("The patient presented with dyspnea.")
    assert [m["role"] for m in messages] == ["system", "user"]
    assert messages[0]["content"] == SYSTEM_PROMPT
    assert messages[1]["content"] == "Sentence: The patient presented with dyspnea."


def test_training_prompt_appends_assistant_target():
    messages = build_chat_messages("Source.", "Target.")
    assert [m["role"] for m in messages] == ["system", "user", "assistant"]
    assert messages[-1]["content"] == "Target."


def test_training_and_generation_share_the_same_prefix():
    # Training labels mask the prompt tokens by assuming the generation prompt is a
    # strict prefix of the training conversation.
    assert build_chat_messages("S", "T")[:2] == build_chat_messages("S")
