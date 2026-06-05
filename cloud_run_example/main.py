"""FastAPI service that runs an ADK agent with the BigQuery Agent Analytics
Plugin attached, shaped for Cloud Run.

Differences from fast_api_example.py at the repo root:

* All configuration comes from environment variables (Cloud Run sets these via
  --set-env-vars on deploy; see cloud_run_example/deploy.sh).
* /health endpoint for Cloud Run's startup / liveness probes.
* PORT environment variable is respected (Cloud Run sets it; default 8080).
* Structured JSON logging via logging.dictConfig so the app log, uvicorn, and
  uvicorn.access all emit JSON for Cloud Logging.
* Lifespan shutdown awaits the plugin's shutdown() so the batch processor
  flushes in-flight rows before the container terminates.

Auth: the container uses Application Default Credentials. On Cloud Run that
means the runtime service account's identity -- no key file is shipped. The
service account needs:
  * roles/bigquery.dataEditor on the target dataset (writes agent_events).
  * roles/bigquery.user on the project (Storage Write API).
  * roles/aiplatform.user on the project (Vertex AI / Gemini).

See deploy.sh for the gcloud commands that wire all of the above up.
"""

from contextlib import asynccontextmanager
import json
import logging
import logging.config
import os
from typing import Any

from fastapi import FastAPI, HTTPException
from google.adk.agents.llm_agent import LlmAgent
from google.adk.apps.app import App
from google.adk.memory.in_memory_memory_service import InMemoryMemoryService
from google.adk.plugins.bigquery_agent_analytics_plugin import (
    BigQueryAgentAnalyticsPlugin,
    BigQueryLoggerConfig,
)
from google.adk.runners import Runner
from google.adk.sessions.in_memory_session_service import InMemorySessionService
from google.genai import types
from pydantic import BaseModel


# --- Structured JSON logging for Cloud Logging ---
# Cloud Logging auto-parses JSON on stdout; emitting structured records makes
# severity, message, and any extra fields queryable in the Logs Explorer.
# Anything passed via `logger.info("...", extra={...})` is included as
# top-level JSON fields.

_STANDARD_LOG_RECORD_ATTRS = frozenset({
    "args", "asctime", "created", "exc_info", "exc_text", "filename",
    "funcName", "levelname", "levelno", "lineno", "message", "module",
    "msecs", "msg", "name", "pathname", "process", "processName",
    "relativeCreated", "stack_info", "taskName", "thread", "threadName",
    "color_message",  # uvicorn's extra
})


class _JsonFormatter(logging.Formatter):
  def format(self, record: logging.LogRecord) -> str:
    payload: dict[str, Any] = {
        "severity": record.levelname,
        "message": record.getMessage(),
        "logger": record.name,
    }
    # Emit any caller-supplied extras (logger.info("...", extra={...}))
    # as top-level fields so they are queryable in Cloud Logging.
    for key, value in record.__dict__.items():
      if key in _STANDARD_LOG_RECORD_ATTRS or key in payload:
        continue
      try:
        json.dumps(value)
        payload[key] = value
      except (TypeError, ValueError):
        payload[key] = repr(value)
    if record.exc_info:
      payload["exception"] = self.formatException(record.exc_info)
    return json.dumps(payload)


# dictConfig fully replaces logging configuration -- including uvicorn's own
# loggers, since uvicorn pulls its formatters via the same root logging
# module. This means the Dockerfile does NOT need --log-config; uvicorn's
# default config (applied at startup) is overridden here at app import time
# because dictConfig sets `disable_existing_loggers=False` and re-binds the
# uvicorn handlers.
_LOG_CONFIG = {
    "version": 1,
    "disable_existing_loggers": False,
    "formatters": {
        "json": {"()": _JsonFormatter},
    },
    "handlers": {
        "stdout": {
            "class": "logging.StreamHandler",
            "stream": "ext://sys.stdout",
            "formatter": "json",
        },
    },
    "root": {"level": "INFO", "handlers": ["stdout"]},
    "loggers": {
        "uvicorn":        {"level": "INFO", "handlers": ["stdout"], "propagate": False},
        "uvicorn.error":  {"level": "INFO", "handlers": ["stdout"], "propagate": False},
        "uvicorn.access": {"level": "INFO", "handlers": ["stdout"], "propagate": False},
    },
}
logging.config.dictConfig(_LOG_CONFIG)
logger = logging.getLogger("bqaa.cloud_run_example")


# --- Configuration (env-var driven for Cloud Run) ---
def _require_env(name: str) -> str:
  value = os.environ.get(name)
  if not value:
    raise RuntimeError(
        f"Required environment variable {name!r} is not set. "
        "Set it via --set-env-vars on `gcloud run deploy` "
        "(see cloud_run_example/deploy.sh)."
    )
  return value


PROJECT_ID = _require_env("GOOGLE_CLOUD_PROJECT")
LOCATION = os.environ.get("GOOGLE_CLOUD_LOCATION", "us-central1")
BQAA_DATASET_ID = _require_env("BQAA_DATASET_ID")
BQAA_TABLE_ID = os.environ.get("BQAA_TABLE_ID", "agent_events")
GEMINI_MODEL = os.environ.get("GEMINI_MODEL", "gemini-2.0-flash")
AGENT_NAME = os.environ.get("AGENT_NAME", "cloud_run_agent")
PORT = int(os.environ.get("PORT", "8080"))

# Vertex AI is the production-friendly path on Cloud Run (no API key
# management; uses the runtime service account's IAM). The Gemini client
# reads these env vars.
os.environ.setdefault("GOOGLE_GENAI_USE_VERTEXAI", "true")
os.environ.setdefault("GOOGLE_CLOUD_LOCATION", LOCATION)


# --- Agent + App + Plugin wiring (matches fast_api_example.py) ---
# LlmAgent.model accepts a model string directly; the ADK registry resolves
# "gemini-*" names to the Gemini provider. No model-class import needed.
agent = LlmAgent(name=AGENT_NAME, model=GEMINI_MODEL)

bq_plugin = BigQueryAgentAnalyticsPlugin(
    project_id=PROJECT_ID,
    dataset_id=BQAA_DATASET_ID,
    table_id=BQAA_TABLE_ID,
    config=BigQueryLoggerConfig(
        log_session_metadata=True,
    ),
)

app_obj = App(
    name="cloud_run_bqaa_app",
    root_agent=agent,
    plugins=[bq_plugin],
)


# --- Session / memory services ---
# In-memory services are fine for the example but DO NOT survive Cloud Run
# instance restarts or scale-to-zero. Replace with FirestoreSessionService
# (or another persistent backend) for any deployment that needs multi-turn
# state across instances.
session_service = InMemorySessionService()
memory_service = InMemoryMemoryService()


# --- FastAPI ---
@asynccontextmanager
async def lifespan(_: FastAPI):
  logger.info(
      "Cloud Run BQAA example starting up",
      extra={
          "project_id": PROJECT_ID,
          "dataset_id": BQAA_DATASET_ID,
          "table_id": BQAA_TABLE_ID,
          "model": GEMINI_MODEL,
      },
  )
  yield
  # Flush in-flight rows before the container terminates. Without this,
  # Cloud Run's SIGTERM grace window may end before the plugin's batch
  # processor has shipped its last batch. The plugin's public lifecycle
  # method is shutdown(); close() exists on internal batch processors only.
  logger.info("Shutting down; flushing BigQuery plugin")
  try:
    if hasattr(bq_plugin, "shutdown"):
      await bq_plugin.shutdown()
    elif hasattr(bq_plugin, "close"):
      await bq_plugin.close()
  except Exception:
    logger.exception("Plugin shutdown raised; events may not have flushed")


app = FastAPI(lifespan=lifespan)


# --- Request / response models ---
class ChatRequest(BaseModel):
  user_id: str
  session_id: str
  message: str


class ChatResponse(BaseModel):
  response: str
  events_logged: int
  session_id: str


# --- Endpoints ---
@app.get("/health")
def health() -> dict:
  """Cloud Run startup / liveness probe target."""
  return {"status": "ok"}


@app.get("/")
def root() -> dict:
  """Root endpoint with a small operator-facing hint."""
  return {
      "service": "bqaa-cloud-run-example",
      "agent": AGENT_NAME,
      "bigquery_target": f"{PROJECT_ID}.{BQAA_DATASET_ID}.{BQAA_TABLE_ID}",
      "endpoints": {"chat": "POST /chat", "health": "GET /health"},
  }


@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest) -> ChatResponse:
  """Run the agent for a single user message; events land in BigQuery."""
  runner = Runner(
      app=app_obj,
      session_service=session_service,
      memory_service=memory_service,
      auto_create_session=True,
  )

  user_msg = types.Content(
      role="user", parts=[types.Part(text=request.message)]
  )

  events = []
  try:
    async for event in runner.run_async(
        user_id=request.user_id,
        session_id=request.session_id,
        new_message=user_msg,
    ):
      events.append(event)

    # Pull the final visible text from the trailing events.
    final_text = "(no text response)"
    for event in reversed(events):
      if event.content and event.content.parts:
        for part in event.content.parts:
          if part.text:
            final_text = part.text
            break
        if final_text != "(no text response)":
          break

    logger.info(
        "Agent turn complete",
        extra={
            "user_id": request.user_id,
            "session_id": request.session_id,
            "events_logged": len(events),
        },
    )
    return ChatResponse(
        response=final_text,
        events_logged=len(events),
        session_id=request.session_id,
    )

  except Exception:
    # Log the full exception server-side; return a generic body to the
    # caller so we don't leak stack traces / model errors / project IDs over
    # an internet-facing endpoint.
    logger.exception(
        "Agent execution failed",
        extra={
            "user_id": request.user_id,
            "session_id": request.session_id,
        },
    )
    raise HTTPException(
        status_code=500, detail="Internal error processing agent turn."
    )


if __name__ == "__main__":
  # Local run: `python main.py`. On Cloud Run, the container's CMD invokes
  # uvicorn directly (see Dockerfile).
  import uvicorn

  uvicorn.run("main:app", host="0.0.0.0", port=PORT, log_config=_LOG_CONFIG)
