"""Smoke tests for the Streamlit demo and the committed artifacts it depends on.

The app only loads the ~1GB base model when "Simplify" is clicked, so rendering the
initial page here downloads nothing.
"""

import json
from pathlib import Path

from streamlit.testing.v1 import AppTest

ROOT = Path(__file__).resolve().parent.parent
ADAPTER_DIR = ROOT / "checkpoints" / "lora_adapter"


def test_streamlit_app_renders():
    at = AppTest.from_file(str(ROOT / "app" / "streamlit_app.py"), default_timeout=120)
    at.run()
    assert not at.exception
    assert len(at.text_area) == 1
    assert at.button[0].label == "Simplify"


def test_empty_input_shows_error_without_loading_model():
    at = AppTest.from_file(str(ROOT / "app" / "streamlit_app.py"), default_timeout=120)
    at.run()
    at.button[0].click().run()
    assert not at.exception
    assert at.error[0].value == "Enter a sentence first."


def test_committed_adapter_matches_documented_config():
    config = json.loads((ADAPTER_DIR / "adapter_config.json").read_text())
    assert config["base_model_name_or_path"] == "Qwen/Qwen2.5-0.5B-Instruct"
    assert config["r"] == 16
    assert config["lora_alpha"] == 32
    assert set(config["target_modules"]) == {"q_proj", "k_proj", "v_proj", "o_proj"}
    assert (ADAPTER_DIR / "adapter_model.safetensors").stat().st_size > 1_000_000


def test_reported_metrics_back_the_readme_claims():
    results = json.loads((ROOT / "reports" / "eval_results.json").read_text())
    base, tuned = results["baseline_zero_shot"], results["fine_tuned"]
    assert base["n_examples"] == tuned["n_examples"] == 200
    for metric in ("sari", "bleu", "rougeL"):
        assert tuned[metric] > base[metric], metric
