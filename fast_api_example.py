from contextlib import asynccontextmanager
import logging
import os
from typing import Any, List, Optional

from fastapi import FastAPI, HTTPException
from google.adk.agents.llm_agent import LlmAgent
# ADK Imports
from google.adk.apps.app import App
from google.adk.memory.in_memory_memory_service import InMemoryMemoryService
from google.adk.models.gemini_model import GeminiModel
from google.adk.plugins.bigquery_agent_analytics_plugin import (
    BigQueryAgentAnalyticsPlugin,
    BigQueryLoggerConfig,
)
from google.adk.runners.runner import Runner
from google.adk.sessions.in_memory_session_service import InMemorySessionService
from google.genai import types
from pydantic import BaseModel

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- Configuration ---
PROJECT_ID = os.environ.get("GOOGLE_CLOUD_PROJECT", "my-project")
DATASET_ID = "my_dataset"
TABLE_ID = "agent_events"

# --- 1. Define your Agent and App ---

# Create a sample agent (using a placeholder model for this example)
model = GeminiModel(model_name="gemini-1.5-flash")
agent = LlmAgent(name="my_agent", model=model)

# Initialize the BigQuery Plugin
# This object holds the configuration and manages the background writer.
bq_plugin = BigQueryAgentAnalyticsPlugin(
    project_id=PROJECT_ID,
    dataset_id=DATASET_ID,
    table_id=TABLE_ID,
    config=BigQueryLoggerConfig(
        log_session_metadata=True,
        # Add any other config options here
    ),
)

# Create the App container, passing the plugin.
# The App is the central registry for plugins and the root agent.
my_app = App(
    name="my_fastapi_app",
    root_agent=agent,
    plugins=[bq_plugin],  # <--- PASS THE PLUGIN HERE
)

# --- 2. Initialize Services ---

# In a real app, you might use persistent services (e.g., FirestoreSessionService)
session_service = InMemorySessionService()
memory_service = InMemoryMemoryService()

# --- 3. FastAPI Setup ---


@asynccontextmanager
async def lifespan(app: FastAPI):
  # Startup logic
  logger.info("Starting up FastAPI app...")
  yield
  # Shutdown logic
  logger.info("Shutting down... verifying plugin cleanup.")
  # The Runner's plugin_manager handles closing plugins, but since we create
  # runners per-request or cached, we might need to ensure plugins are closed
  # if they hold global state.
  # However, BigQueryAgentAnalyticsPlugin is attached to the App.
  # The 'App' object itself doesn't track runtime state, the Runner does.
  # Note: If you share the `bq_plugin` instance across runners (which is typical
  # for `App` based usage), strict cleanup might require manual intervention
  # if the Runner instances are ephemeral.
  # But `BigQueryAgentAnalyticsPlugin` uses `atexit` and weakrefs to try to cleanup.
  # For robust graceful shutdown, you might want to explicitly close it if you
  # managing the lifecycle manually.
  if hasattr(bq_plugin, "close"):
    await bq_plugin.close()


app = FastAPI(lifespan=lifespan)

# --- 4. Request Models ---


class ChatRequest(BaseModel):
  user_id: str
  session_id: str
  message: str


class ChatResponse(BaseModel):
  response: str
  events_logged: int


# --- 5. Define Routes ---


@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest):
  """Executes the agent for a given user message and logs events to BigQuery."""

  # Create the Runner for this request.
  # We pass the `my_app` which already contains the `bq_plugin`.
  runner = Runner(
      app=my_app,
      session_service=session_service,
      memory_service=memory_service,
      auto_create_session=True,  # Simplify session creation for this example
  )

  # Convert string message to ADK Content
  user_msg = types.Content(
      role="user", parts=[types.Part(text=request.message)]
  )

  events = []
  try:
    # Execute the agent asynchronously
    async for event in runner.run_async(
        user_id=request.user_id,
        session_id=request.session_id,
        new_message=user_msg,
    ):
      events.append(event)
      # You could stream these events back to the client using SSE
      # (Server-Sent Events) if desired.

    # Extract the final model response text for the API response
    final_text = "No response"
    for event in reversed(events):
      if event.content and event.content.parts:
        # simple heuristic to find the last text part
        for part in event.content.parts:
          if part.text:
            final_text = part.text
            break
        if final_text != "No response":
          break

    return ChatResponse(response=final_text, events_logged=len(events))

  except Exception as e:
    logger.error(f"Error during agent execution: {e}")
    raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
  import uvicorn
  # Run with: python fastapi_integration.py
  uvicorn.run(app, host="0.0.0.0", port=8000)
