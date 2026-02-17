"""Local multi-agent system for testing BQ Agent Analytics Plugin.

Architecture (3 levels):
  root_agent (orchestrator)
    +-- data_team (SequentialAgent)   <- Level 2
    |     +-- schema_explorer        <- Level 3 (LlmAgent)
    |     +-- query_analyst          <- Level 3 (LlmAgent)
    +-- image_describer              <- Level 2 (LlmAgent, multimodal)
"""

from google.adk.agents import LlmAgent
from google.adk.agents.sequential_agent import SequentialAgent
from google.adk.apps.app import App
from google.adk.plugins.bigquery_agent_analytics_plugin import (
    BigQueryAgentAnalyticsPlugin,
    BigQueryLoggerConfig,
)
from google.adk.plugins.multimodal_tool_results_plugin import (
    MultimodalToolResultsPlugin,
)
from google.adk.tools.bigquery.bigquery_credentials import (
    BigQueryCredentialsConfig,
)
from google.adk.tools.bigquery.bigquery_toolset import BigQueryToolset
from google.adk.tools.bigquery.config import BigQueryToolConfig, WriteMode
from google.genai import types
import google.auth

MODEL = "gemini-2.5-flash"
# TODO: Replace with your own project ID
PROJECT_ID = "YOUR_PROJECT_ID"
DATASET_ID = "agent_analytics"

# ---------- Credentials & BQ Toolset ----------
adc, _ = google.auth.default()
bq_creds = BigQueryCredentialsConfig(credentials=adc)
bq_config = BigQueryToolConfig(
    write_mode=WriteMode.BLOCKED,
    application_name="bq_analytics_test_local",
)
bq_toolset = BigQueryToolset(
    credentials_config=bq_creds,
    bigquery_tool_config=bq_config,
)

# ---------- Multimodal tool ----------
def describe_this_image(image_uri: str):
    """Fetches an image from a GCS or HTTPS URI for the LLM to describe.

    Args:
        image_uri: A gs:// or https:// URI pointing to an image.
    """
    return [types.Part.from_uri(file_uri=image_uri, mime_type="image/jpeg")]


# ---------- Level-3 agents ----------
schema_explorer = LlmAgent(
    name="schema_explorer",
    model=MODEL,
    instruction=(
        "You are a BigQuery Schema Explorer. "
        "Use list_dataset_ids and list_table_ids and get_table_info "
        "to help users understand what data is available. "
        "Summarise the schema clearly."
    ),
    description="Explores BigQuery schemas, datasets, and table structures.",
    tools=[bq_toolset],
    output_key="schema_summary",
)

query_analyst = LlmAgent(
    name="query_analyst",
    model=MODEL,
    instruction=(
        "You are a BigQuery Query Analyst. "
        "Given the schema summary below, write and execute SQL to answer "
        "the user's question.\n\n"
        "Schema summary:\n{schema_summary}\n\n"
        "Use execute_sql to run queries. Explain results clearly."
    ),
    description="Writes and runs SQL queries based on schema context.",
    tools=[bq_toolset],
    output_key="query_result",
)

# ---------- Level-2 agents ----------
data_team = SequentialAgent(
    name="data_team",
    sub_agents=[schema_explorer, query_analyst],
    description=(
        "A sequential pipeline: first explores schema, then runs queries."
    ),
)

image_describer = LlmAgent(
    name="image_describer",
    model=MODEL,
    instruction=(
        "You are an Image Description Agent. "
        "When asked to describe an image, use the describe_this_image tool "
        "to fetch it, then describe what you see in detail."
    ),
    description="Describes images from GCS or HTTPS URIs.",
    tools=[describe_this_image],
)

# ---------- Level-1: Root orchestrator ----------
root_agent = LlmAgent(
    name="orchestrator",
    model=MODEL,
    instruction=(
        "You are the main orchestrator. You have two teams:\n"
        "1. **data_team** -- for BigQuery schema exploration and SQL queries. "
        "Transfer to data_team when the user asks about data, tables, "
        "or wants to run queries.\n"
        "2. **image_describer** -- for describing images. Transfer to "
        "image_describer when the user provides an image URI.\n\n"
        "For simple greetings or general questions, answer directly."
    ),
    description="Root orchestrator that delegates to specialist teams.",
    sub_agents=[data_team, image_describer],
)

# ---------- App with BQ Analytics Plugin ----------
bq_analytics_plugin = BigQueryAgentAnalyticsPlugin(
    project_id=PROJECT_ID,
    dataset_id=DATASET_ID,
    config=BigQueryLoggerConfig(
        batch_size=1,
        batch_flush_interval=0.5,
        log_session_metadata=True,
        custom_tags={"env": "local_test", "test": "eventdata_refactor"},
    ),
)

app = App(
    name="bq_analytics_test_local",
    root_agent=root_agent,
    plugins=[bq_analytics_plugin, MultimodalToolResultsPlugin()],
)
