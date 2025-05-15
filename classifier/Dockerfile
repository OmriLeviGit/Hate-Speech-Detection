FROM python:3.9-slim AS builder

# Add build argument with CPU as default
ARG PYTORCH_URL="https://download.pytorch.org/whl/cpu"
ARG TARGET="cpu"

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt /app/

# Use the build argument for PyTorch URL, use a bash trick to avoid rebuilding every time
RUN bash -c "echo \"--extra-index-url ${PYTORCH_URL}\" > /tmp/reqs && \
    cat /app/requirements.txt >> /tmp/reqs && \
    pip install --prefer-binary -r /tmp/reqs"

RUN python -m spacy download en_core_web_lg
RUN python -m nltk.downloader wordnet omw-1.4 averaged_perceptron_tagger averaged_perceptron_tagger_eng -d /usr/local/share/nltk_data

COPY src /app/classifier/

FROM python:3.9-slim

COPY --from=builder /usr/local /usr/local
COPY --from=builder /app /app

CMD ["python3", "-m", "classifier.src.compare_models"]