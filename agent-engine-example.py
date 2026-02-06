import os
import shutil
from google.adk.agents import Agent
from google.adk.models.google_llm import Gemini
from google.adk.plugins.bigquery_agent_analytics_plugin import BigQueryAgentAnalyticsPlugin
from google.adk.tools.bigquery import BigQueryCredentialsConfig, BigQueryToolset
from google.adk.tools.tool_context import ToolContext
import google.auth
import vertexai
from vertexai import agent_engines
# --- Configuration ---
PROJECT_ID = "<your project>"
LOCATION = "<your location>"
STAGING_BUCKET = "gs://<your bucket>"
# ... [Checks for env vars] ...
os.environ["GOOGLE_GENAI_USE_VERTEXAI"] = "True"
BQ_DATASET_ID = "<your dataset>"
BQ_TABLE_ID = "<your table>"

vertexai.init(
    project=PROJECT_ID, location=LOCATION, staging_bucket=STAGING_BUCKET
)
client = vertexai.Client(project=PROJECT_ID, location=LOCATION)
bq_logger_plugin = BigQueryAgentAnalyticsPlugin(
    project_id=PROJECT_ID,
    dataset_id=BQ_DATASET_ID,
    table_id=BQ_TABLE_ID,
    log_session_metadata=True,
    custom_tags={
        "agent_role": "sales_ae",
        "env": "agent_engine",
        "version": "v8_state_delta",
    },
)
try:
  credentials, _ = google.auth.default(
      scopes=["https://www.googleapis.com/auth/cloud-platform"]
  )
except google.auth.exceptions.DefaultCredentialsError:
  raise RuntimeError("Could not get default credentials.")
bq_creds_config = BigQueryCredentialsConfig(credentials=credentials)
bigquery_toolset = BigQueryToolset(credentials_config=bq_creds_config)
llm = Gemini(model="gemini-2.5-flash")


def set_state(key: str, value: str, tool_context: ToolContext) -> str:
  """Sets a key-value pair in the session state."""
  tool_context.state[key] = value
  return f"Set state {key} to {value}"


root_agent = Agent(
    model=llm,
    name="my_bq_agent",
    instruction=(
        "You are a helpful assistant. You can use BigQuery tools to answer"
        " questions about data. You can also set session state using the"
        " `set_state` tool."
    ),
    tools=[bigquery_toolset, set_state],
)
app_to_deploy = agent_engines.AdkApp(
    agent=root_agent, plugins=[bq_logger_plugin]
)

# --- 1. Prepare Local Dependencies Directory ---
# Define paths
local_whl_source = "<your path>/google_adk-1.24.0-py3-none-any.whl"
dep_dir = "./adk_dependencies"  # Local temporary directory
whl_basename = os.path.basename(local_whl_source)
local_whl_dest = os.path.join(dep_dir, whl_basename)
# Create directory and copy file
if not os.path.exists(local_whl_source):
  raise FileNotFoundError(f"Source wheel not found: {local_whl_source}")
if os.path.exists(dep_dir):
  shutil.rmtree(dep_dir)  # Clean up previous runs
os.makedirs(dep_dir)
shutil.copy(local_whl_source, local_whl_dest)
print(f"Prepared local dependency: {local_whl_dest}")
# --- 2. Deploy ---
print(f"Deploying agent to Vertex AI Agent Engine...")
try:
  remote_app = client.agent_engines.create(
      agent=app_to_deploy,
      config={
          "display_name": "<your name>",
          "staging_bucket": STAGING_BUCKET,
          "requirements": [
              "google-cloud-aiplatform[agent_engines]",
              # Reference the file inside the uploaded directory
              f"adk_dependencies/{whl_basename}",
              "google-cloud-bigquery",
              "google-auth",
              "db-dtypes",
              "pyarrow",
              "pydantic",
              "cloudpickle",
          ],
          # Upload the entire directory. It should appear as /code/adk_dependencies/ remotely.
          "extra_packages": [dep_dir],
      },
  )
  print(f"Deployed agent: {remote_app.api_resource.name}")
except Exception as e:
  print(f"Failed to deploy agent: {e}")
  import traceback

  traceback.print_exc()
finally:
  # Optional: Cleanup the temporary directory
  # if os.path.exists(dep_dir):
  #    shutil.rmtree(dep_dir)
  pass
