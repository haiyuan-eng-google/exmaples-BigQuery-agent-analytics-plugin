# Cloud Run-ready container for the BQAA + ADK example.
#
# Build locally:
#   docker build -t bqaa-cloud-run-example cloud_run_example/
# Or let `gcloud run deploy --source cloud_run_example` invoke Cloud Build.
FROM python:3.11-slim

# Faster, more reproducible Python in a container.
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

# Install dependencies first so the layer cache is reused across code edits.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the application source.
COPY main.py .

# Cloud Run sets PORT; default to 8080 for local runs.
ENV PORT=8080
EXPOSE 8080

# uvicorn binds 0.0.0.0 so Cloud Run can reach it. log-config=none keeps
# uvicorn from overriding the JSON formatter the app installs.
CMD ["sh", "-c", "uvicorn main:app --host 0.0.0.0 --port ${PORT:-8080} --log-config /dev/null"]
