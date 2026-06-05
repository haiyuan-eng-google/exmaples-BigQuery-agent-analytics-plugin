# Cloud Run example

Deploy the ADK BigQuery Agent Analytics Plugin behind a FastAPI service on
Cloud Run. The agent shape matches `fast_api_example.py` at the repo root
(single `LlmAgent` with the plugin attached via `App`); this directory adds
the container + IAM + deploy plumbing.

## What you get

- `main.py` — FastAPI service: `/chat` (run the agent), `/health` (Cloud Run
  probe), `/` (small operator hint). All config from env vars. Structured JSON
  logging so Cloud Logging parses severity. Lifespan shutdown flushes the
  plugin's batch processor before SIGTERM.
- `Dockerfile` — Python 3.11 slim base; `uvicorn` on `PORT` (default 8080).
- `requirements.txt` — `google-adk>=1.26.0` (the BQAA plugin floor),
  FastAPI/uvicorn, BigQuery Storage Write API path (PyArrow).
- `deploy.sh` — End-to-end deploy: enable APIs, create dataset, create runtime
  service account, grant IAM, `gcloud run deploy --source`. Idempotent.
- `.dockerignore` — Keeps `deploy.sh` and the markdown out of the image.

## Prerequisites

- `gcloud` and `bq` CLIs installed and authenticated.
- A Google Cloud project with billing enabled.
- A region (default `us-central1`).

## Deploy

```bash
PROJECT_ID=your-project \
REGION=us-central1 \
BQAA_DATASET_ID=agent_analytics \
  bash cloud_run_example/deploy.sh
```

The script:

1. Enables `run`, `cloudbuild`, `artifactregistry`, `bigquery`,
   `bigquerystorage`, `aiplatform`, and `iam` APIs.
2. Creates the BigQuery dataset if it doesn't exist (region-matched).
3. Creates a runtime service account (`bqaa-cloud-run-sa@…`).
4. Grants:
   - `roles/bigquery.dataEditor` on the dataset (writes `agent_events`).
   - `roles/bigquery.user` on the project (Storage Write API).
   - `roles/aiplatform.user` on the project (Gemini via Vertex AI).
5. Builds the container via Cloud Build (`gcloud run deploy --source`).
6. Deploys to Cloud Run with the env vars `main.py` expects.
7. Prints the invoke URL and a sample `curl` + BigQuery query.

## Smoke test after deploy

The deploy output ends with the exact commands. The short form:

```bash
URL=https://your-service-xxxx-uc.a.run.app

curl -s -X POST "$URL/chat" \
  -H 'Content-Type: application/json' \
  -d '{"user_id":"u1","session_id":"s1","message":"Hello"}' | jq

bq query --project_id="$PROJECT_ID" --use_legacy_sql=false "
  SELECT event_type, agent, COUNT(*) AS n
    FROM \`$PROJECT_ID.$BQAA_DATASET_ID.agent_events\`
    WHERE timestamp >= TIMESTAMP_SUB(CURRENT_TIMESTAMP(), INTERVAL 1 HOUR)
    GROUP BY event_type, agent
    ORDER BY n DESC"
```

You should see at least `USER_MESSAGE_RECEIVED`, `AGENT_STARTING`,
`LLM_REQUEST`, `LLM_RESPONSE`, `AGENT_RESPONSE`, `AGENT_COMPLETED` rows.

## Configuration (env vars)

| Variable | Required? | Default | Purpose |
|---|---|---|---|
| `GOOGLE_CLOUD_PROJECT` | yes | — | BigQuery + Vertex AI project. |
| `BQAA_DATASET_ID` | yes | — | BigQuery dataset receiving `agent_events`. |
| `GOOGLE_CLOUD_LOCATION` | no | `us-central1` | Vertex AI region. |
| `BQAA_TABLE_ID` | no | `agent_events` | Target table name. |
| `GEMINI_MODEL` | no | `gemini-2.0-flash` | LLM. |
| `AGENT_NAME` | no | `cloud_run_agent` | The agent's `name` attribute (also stamped onto `agent_events.agent`). |
| `PORT` | no | `8080` | Cloud Run sets this; respect it. |
| `GOOGLE_GENAI_USE_VERTEXAI` | no | auto-set to `true` | Routes the Gemini client through Vertex AI (no API key needed; uses the runtime SA). |

## Local run

```bash
cd cloud_run_example
pip install -r requirements.txt
GOOGLE_CLOUD_PROJECT=your-project \
GOOGLE_CLOUD_LOCATION=us-central1 \
BQAA_DATASET_ID=agent_analytics \
  python main.py
```

Then hit `http://localhost:8080/chat` with the same curl as the deploy
smoke test. Local runs use Application Default Credentials, so run
`gcloud auth application-default login` first.

## Production notes

- **Session state is in-memory.** `InMemorySessionService` does not survive
  Cloud Run instance restarts or scale-to-zero. For multi-turn conversations
  that must persist across instances, replace with a persistent session
  service (e.g. Firestore-backed). Two-line change in `main.py`.
- **Graceful shutdown matters.** `lifespan` awaits `bq_plugin.close()` so
  in-flight rows flush before SIGTERM. Cloud Run's default shutdown grace
  window is 10 seconds; if you increase the plugin's `batch_flush_interval`
  significantly, increase Cloud Run's `--no-cpu-throttling` + termination
  grace too.
- **IAM scope.** `deploy.sh` grants `roles/bigquery.dataEditor` at the
  **dataset** level (least privilege for writes). The project-level
  `roles/bigquery.user` grant is required for Storage Write API jobs and
  cannot be scoped tighter today.
- **Unauthenticated invocation.** `deploy.sh` uses `--allow-unauthenticated`
  for the smoke test. For anything not throwaway, drop that flag and put
  Identity-Aware Proxy or your own auth layer in front of the service.
- **Cost.** Each Cloud Run cold start runs the plugin's initial table
  create-if-missing call. The Vertex AI / Gemini call cost dominates the
  per-request bill; BigQuery Storage Write API is sub-cent per request at
  example volumes.

## Verifying the plugin from BigQuery

```sql
-- Per-event-type volume in the last hour.
SELECT event_type, agent, COUNT(*) AS n
FROM `your-project.agent_analytics.agent_events`
WHERE timestamp >= TIMESTAMP_SUB(CURRENT_TIMESTAMP(), INTERVAL 1 HOUR)
GROUP BY event_type, agent
ORDER BY n DESC;

-- Recent sessions with their visible response text.
SELECT
  session_id,
  user_id,
  timestamp,
  JSON_VALUE(content, '$.response') AS visible_response
FROM `your-project.agent_analytics.agent_events`
WHERE event_type = 'AGENT_RESPONSE'
  AND timestamp >= TIMESTAMP_SUB(CURRENT_TIMESTAMP(), INTERVAL 1 HOUR)
ORDER BY timestamp DESC
LIMIT 20;
```

## How this differs from `fast_api_example.py`

`fast_api_example.py` at the repo root is the same wiring without the Cloud
Run-specific pieces. If you've already read that file, the deltas are:

- Env-var-driven config (Cloud Run's `--set-env-vars`) instead of inline
  Python constants.
- Required-env-var guards with helpful error messages so misconfigured
  deploys fail fast at startup, not on the first request.
- `/health` endpoint for Cloud Run startup/liveness probes.
- Structured JSON logging so Cloud Logging picks up severity.
- Dockerfile + `.dockerignore` + `deploy.sh` for `gcloud run deploy --source`.
- IAM matrix grants the runtime service account the right roles instead of
  relying on local ADC.
