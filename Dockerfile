FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src/ src/
COPY app/ app/
COPY checkpoints/lora_adapter/ checkpoints/lora_adapter/

# Bake the base model into the image so containers start instantly instead of
# pulling ~1GB from the HF Hub on every cold start -- anonymous Hub downloads
# are rate-limited and measured at ~14 minutes from inside a fresh container.
RUN python -c "from transformers import AutoModelForCausalLM, AutoTokenizer; \
    AutoModelForCausalLM.from_pretrained('Qwen/Qwen2.5-0.5B-Instruct'); \
    AutoTokenizer.from_pretrained('Qwen/Qwen2.5-0.5B-Instruct')"

# Without this, from_pretrained still does a network round-trip to validate the
# cached model against the Hub on every load, even though the weights are
# already baked in above -- slow and pointless once the image is built.
ENV HF_HUB_OFFLINE=1

EXPOSE 8501

HEALTHCHECK CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8501/_stcore/health')" || exit 1

CMD ["streamlit", "run", "app/streamlit_app.py", "--server.port=8501", "--server.address=0.0.0.0", "--server.fileWatcherType=none"]
