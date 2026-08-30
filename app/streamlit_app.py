"""
Interactive demo: paste a biomedical/clinical sentence and compare the base
Qwen2.5-0.5B-Instruct model's zero-shot simplification against the same
model with the project's LoRA adapter applied.

Loads ONE base model and toggles the LoRA adapter on/off with PEFT's
`disable_adapter()` context manager, rather than holding two full model
copies in memory -- this machine (and Streamlit Community Cloud's free
tier) has hit real OOM limits on past projects, so this app follows the
same single-model-instance discipline `src/evaluate_model.py` uses.
"""
import contextlib
import sys
from pathlib import Path

import streamlit as st
import textstat
import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))
from prompting import build_chat_messages  # noqa: E402

MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"
ADAPTER_DIR = PROJECT_ROOT / "checkpoints" / "lora_adapter"


@st.cache_resource(show_spinner="Loading Qwen2.5-0.5B-Instruct (first run only, ~1GB download)...")
def load_model():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    base_model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, torch_dtype=torch.float32, low_cpu_mem_usage=True
    )
    model = PeftModel.from_pretrained(base_model, str(ADAPTER_DIR))
    model.eval()
    return tokenizer, model


def generate(tokenizer, model, source: str, use_adapter: bool, max_new_tokens: int = 100) -> str:
    messages = build_chat_messages(source, None)
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors="pt")
    ctx = contextlib.nullcontext() if use_adapter else model.disable_adapter()
    with torch.no_grad(), ctx:
        out = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            num_beams=1,
            temperature=None,
            top_p=None,
            pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
        )
    gen_ids = out[0][inputs["input_ids"].shape[1]:]
    result = tokenizer.decode(gen_ids, skip_special_tokens=True).strip()
    return result.split("\n")[0].strip()


def fk_grade(text: str) -> float:
    return textstat.flesch_kincaid_grade(text) if text.strip() else 0.0


st.set_page_config(page_title="Clinical Text Simplifier", page_icon="\U0001FA7A", layout="wide")

st.title("Clinical Text Simplifier")
st.markdown(
    "LoRA fine-tuning of a 0.5B-parameter open model "
    "([`Qwen2.5-0.5B-Instruct`](https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct)) "
    "to rewrite biomedical sentences in plain language, evaluated against "
    "[PLABA](https://bionlp.nlm.nih.gov/plaba2024/), a real NIH/NLM dataset of "
    "professionally-written plain-language simplifications. "
    "[Full case study](https://github.com/crisomara/clinical-text-simplifier/blob/master/reports/case_study.md) "
    "&middot; [source](https://github.com/crisomara/clinical-text-simplifier)"
)

st.warning(
    "**Research demo, not a medical device.** This is a 0.5B-parameter model fine-tuned "
    "on 800 examples for a portfolio case study. It has a documented failure mode where "
    "it drops or invents clinical details (see the case study for real examples) -- do "
    "not use its output for real patient care or clinical decisions.",
    icon="⚠️",
)

source_text = st.text_area(
    "Biomedical sentence to simplify",
    height=100,
    placeholder="e.g. The patient presented with acute exacerbation of chronic obstructive pulmonary disease.",
    label_visibility="collapsed",
)

run_live = st.button("Simplify", type="primary")

if run_live:
    if not source_text.strip():
        st.error("Enter a sentence first.")
    else:
        tokenizer, model = load_model()
        with st.spinner("Generating (CPU inference, a few seconds per model)..."):
            baseline_out = generate(tokenizer, model, source_text, use_adapter=False)
            fine_tuned_out = generate(tokenizer, model, source_text, use_adapter=True)

        left, right = st.columns(2)
        with left:
            st.markdown("**Zero-shot baseline**")
            st.info(baseline_out)
            st.caption(f"Flesch-Kincaid grade: {fk_grade(baseline_out):.1f}")
        with right:
            st.markdown("**LoRA fine-tuned**")
            st.success(fine_tuned_out)
            st.caption(f"Flesch-Kincaid grade: {fk_grade(fine_tuned_out):.1f}")
        st.caption(f"Source Flesch-Kincaid grade: {fk_grade(source_text):.1f}")

st.divider()
st.subheader("Held-out test results (200 PLABA sentences, never seen in training)")
metric_cols = st.columns(4)
metric_cols[0].metric("SARI", "34.11", "+5.06 vs. baseline")
metric_cols[1].metric("sacreBLEU", "17.40", "+7.92 vs. baseline (+83%)")
metric_cols[2].metric("ROUGE-L", "0.380", "+0.084 vs. baseline")
metric_cols[3].metric("FK grade delta", "-3.51", "vs. -2.36 baseline")
st.caption(
    "The fine-tune improved every metric measured on held-out data -- but it also "
    "simplifies more aggressively than human annotators do and has real failure modes "
    "(dropped clinical content, an invented number). Full methodology, limitations, and "
    "all 200 scored examples in the case study."
)
