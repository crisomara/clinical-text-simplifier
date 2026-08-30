"""
Interactive demo: paste a biomedical/clinical sentence and compare the base
Qwen2.5-0.5B-Instruct model's zero-shot simplification against the same
model with the project's LoRA adapter applied.

Loads ONE base model and toggles the LoRA adapter on/off with PEFT's
`disable_adapter()` context manager, rather than holding two full model
copies in memory -- this machine (and Streamlit Community Cloud's free
tier) has hit real OOM limits on past projects, so this app follows the
same single-model-instance discipline `src/evaluate_model.py` uses.

Visual design mirrors docs/index.html (the project's GitHub Pages case
study page) so the demo and the write-up read as one piece: same paper/ink
palette, Fraunces/Inter/IBM Plex Mono type system, and "case-note" card
language for baseline-vs-fine-tuned results. Uses st.html() (not
st.markdown(unsafe_allow_html=True)) for anything with real markup --
markdown's raw-HTML-block passthrough silently breaks on blank lines,
which a multi-rule <style> block always has.
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

CSS = """
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Fraunces:opsz,wght@9..144,400;9..144,500;9..144,600;9..144,700&family=Inter:wght@400;500;600;700&family=IBM+Plex+Mono:wght@500;600&display=swap" rel="stylesheet">
<style>
  :root {
    --ink: #1B1D1B; --paper: #F1F3EF; --paper-raised: #E7EAE3; --paper-raised-2: #DDE1D6;
    --accent: #B5652A; --accent-2: #3E7C74; --rule: #CDD1C6; --muted: #5B5F58; --muted-2: #767B72;
    --bad: #9C4A34;
    --font-display: "Fraunces", Georgia, "Times New Roman", serif;
    --font-body: "Inter", -apple-system, "Segoe UI", sans-serif;
    --font-mono: "IBM Plex Mono", "SFMono-Regular", Consolas, monospace;
  }
  html, body, [class*="css"] { font-family: var(--font-body); }
  .stApp { background: var(--paper); color: var(--ink); }
  .block-container { max-width: 780px; padding-top: 3.5rem; padding-bottom: 4rem; }
  .eyebrow { font-family: var(--font-mono); font-size: 0.72rem; letter-spacing: 0.14em;
             text-transform: uppercase; color: var(--accent-2); margin: 0 0 1rem; }
  h1.headline { font-family: var(--font-display); font-optical-sizing: auto; font-weight: 600;
                font-size: clamp(2.1rem, 5vw, 2.9rem); line-height: 1.08; letter-spacing: -0.01em;
                margin: 0 0 1.1rem; color: var(--ink); }
  .dek { font-size: 1.05rem; color: var(--muted); max-width: 62ch; margin: 0 0 0.5rem; line-height: 1.6; }
  .dek a { color: var(--accent); text-decoration: none; border-bottom: 1px solid currentColor; }
  .dek a:hover { color: var(--accent-2); }
  .dek code { font-family: var(--font-mono); font-size: 0.86em; background: var(--paper-raised);
              padding: 0.1em 0.35em; border-radius: 3px; }
  .stat-row { display: flex; gap: 2.2rem; flex-wrap: wrap; margin: 2rem 0 2.4rem;
              padding-top: 1.6rem; border-top: 1px solid var(--rule); }
  .stat { min-width: 110px; }
  .stat .n { font-family: var(--font-mono); font-weight: 600; font-size: 1.5rem; color: var(--accent);
             font-variant-numeric: tabular-nums; display: block; }
  .stat .d { font-size: 0.76rem; color: var(--muted); margin-top: 0.15rem; }
  .callout.warn { background: var(--paper-raised); border-left: 3px solid var(--bad);
                  padding: 1rem 1.3rem; margin: 0 0 2rem; border-radius: 0 4px 4px 0;
                  font-size: 0.92rem; line-height: 1.55; color: var(--ink); }
  .callout.warn strong { color: var(--bad); }
  .section-label { font-family: var(--font-mono); font-size: 0.7rem; letter-spacing: 0.1em;
                    text-transform: uppercase; color: var(--muted-2); margin: 0 0 0.7rem; }
  div[data-testid="stTextArea"] textarea { font-family: var(--font-body); background: #FFFFFF;
                    border: 1px solid var(--rule); border-radius: 6px; color: var(--ink); }
  div[data-testid="stTextArea"] textarea:focus { border-color: var(--accent-2); box-shadow: none; }
  div.stButton > button { font-family: var(--font-mono); font-size: 0.82rem; letter-spacing: 0.03em;
                    text-transform: uppercase; background: var(--ink); color: var(--paper);
                    border: none; border-radius: 6px; padding: 0.55rem 1.4rem; }
  div.stButton > button:hover { background: var(--accent); color: #fff; }
  div.stButton > button:focus:not(:active) { border-color: var(--accent); color: var(--paper); }
  .case-note { border: 1px solid var(--rule); border-radius: 6px; padding: 1.1rem 1.3rem;
               margin-bottom: 0; height: 100%; background: #FFFFFF; }
  .case-note .verdict { font-family: var(--font-mono); font-size: 0.64rem; letter-spacing: 0.08em;
                         text-transform: uppercase; padding: 0.15rem 0.55rem; border-radius: 2px;
                         display: inline-block; margin-bottom: 0.8rem; background: var(--paper-raised); }
  .case-note.tuned .verdict { color: var(--accent-2); }
  .case-note.base .verdict { color: var(--muted); }
  .case-note p { font-size: 0.94rem; line-height: 1.55; margin: 0 0 0.8rem; color: var(--ink); }
  .case-note .fk { font-family: var(--font-mono); font-size: 0.72rem; color: var(--muted-2);
                    text-transform: uppercase; letter-spacing: 0.04em; }
  .links { display: flex; gap: 1.3rem; flex-wrap: wrap; margin-top: 3rem; padding-top: 1.6rem;
           border-top: 1px solid var(--rule); }
  .links a { font-family: var(--font-mono); font-size: 0.8rem; color: var(--accent);
             text-decoration: none; border-bottom: 1px solid currentColor; padding-bottom: 1px; }
  .links a:hover { color: var(--accent-2); }
</style>
"""

HERO = """
<p class="eyebrow">Research demo &middot; LoRA fine-tuning</p>
<h1 class="headline">Clinical Text Simplifier</h1>
<p class="dek">A 0.5B-parameter open model
(<code>Qwen2.5-0.5B-Instruct</code>) fine-tuned to rewrite biomedical sentences
in plain language, evaluated against
<a href="https://bionlp.nlm.nih.gov/plaba2024/" target="_blank">PLABA</a>, a real
NIH/NLM dataset of professionally-written plain-language simplifications.</p>
<div class="stat-row">
  <div class="stat"><span class="n">34.11</span><span class="d">SARI &middot; +5.06</span></div>
  <div class="stat"><span class="n">17.40</span><span class="d">sacreBLEU &middot; +7.92</span></div>
  <div class="stat"><span class="n">0.380</span><span class="d">ROUGE-L &middot; +0.084</span></div>
  <div class="stat"><span class="n">&minus;3.51</span><span class="d">FK grade &middot; vs &minus;2.36</span></div>
</div>
<div class="callout warn"><strong>Research demo, not a medical device.</strong>
This model has a documented failure mode where it drops or invents clinical
details (see the case study for real examples) -- do not use its output for
real patient care or clinical decisions.</div>
<p class="section-label">Try it</p>
"""

FOOTER = """
<div class="links">
  <a href="https://github.com/crisomara/clinical-text-simplifier/blob/master/reports/case_study.md" target="_blank">Full case study &rarr;</a>
  <a href="https://github.com/crisomara/clinical-text-simplifier" target="_blank">Source &rarr;</a>
  <a href="https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct" target="_blank">Base model &rarr;</a>
</div>
"""


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


st.set_page_config(page_title="Clinical Text Simplifier", page_icon="\U0001FA7A", layout="centered")

st.html(CSS)
st.html(HERO)

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
            st.html(
                f'<div class="case-note base"><div class="verdict">Zero-shot baseline</div>'
                f'<p>{baseline_out}</p>'
                f'<div class="fk">FK grade {fk_grade(baseline_out):.1f}</div></div>'
            )
        with right:
            st.html(
                f'<div class="case-note tuned"><div class="verdict">LoRA fine-tuned</div>'
                f'<p>{fine_tuned_out}</p>'
                f'<div class="fk">FK grade {fk_grade(fine_tuned_out):.1f}</div></div>'
            )

st.html(FOOTER)
