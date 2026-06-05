#!/usr/bin/env bash
# Deploys the BQAA + ADK Cloud Run example end-to-end.
#
# Usage:
#   PROJECT_ID=my-project \
#   REGION=us-central1 \
#   BQAA_DATASET_ID=agent_analytics \
#     bash cloud_run_example/deploy.sh
#
# Optional overrides:
#   SERVICE_NAME      (default: bqaa-cloud-run-example)
#   SERVICE_ACCOUNT   (default: bqaa-cloud-run-sa)
#   BQAA_TABLE_ID     (default: agent_events)
#   GEMINI_MODEL      (default: gemini-2.0-flash)
#   AGENT_NAME        (default: cloud_run_agent)
#
# The script is idempotent where possible (re-running is safe).

set -euo pipefail

PROJECT_ID="${PROJECT_ID:?PROJECT_ID is required}"
REGION="${REGION:-us-central1}"
BQAA_DATASET_ID="${BQAA_DATASET_ID:?BQAA_DATASET_ID is required (BigQuery dataset for agent_events)}"
BQAA_TABLE_ID="${BQAA_TABLE_ID:-agent_events}"
SERVICE_NAME="${SERVICE_NAME:-bqaa-cloud-run-example}"
SERVICE_ACCOUNT="${SERVICE_ACCOUNT:-bqaa-cloud-run-sa}"
GEMINI_MODEL="${GEMINI_MODEL:-gemini-2.0-flash}"
AGENT_NAME="${AGENT_NAME:-cloud_run_agent}"

SA_EMAIL="${SERVICE_ACCOUNT}@${PROJECT_ID}.iam.gserviceaccount.com"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo ">>> Project:         ${PROJECT_ID}"
echo ">>> Region:          ${REGION}"
echo ">>> Service:         ${SERVICE_NAME}"
echo ">>> Service account: ${SA_EMAIL}"
echo ">>> BigQuery target: ${PROJECT_ID}.${BQAA_DATASET_ID}.${BQAA_TABLE_ID}"
echo ">>> Model:           ${GEMINI_MODEL}"
echo

# 1. Enable the APIs the deploy needs.
echo "--- Enabling required APIs ---"
gcloud services enable \
  run.googleapis.com \
  cloudbuild.googleapis.com \
  artifactregistry.googleapis.com \
  bigquery.googleapis.com \
  bigquerystorage.googleapis.com \
  aiplatform.googleapis.com \
  iam.googleapis.com \
  --project="${PROJECT_ID}"

# 2. Create the BigQuery dataset if missing. Location matches the Cloud Run
#    region's BQ region (us-central1 -> US multi-region).
echo
echo "--- Ensuring BigQuery dataset exists ---"
if ! bq --project_id="${PROJECT_ID}" show "${PROJECT_ID}:${BQAA_DATASET_ID}" >/dev/null 2>&1; then
  bq --project_id="${PROJECT_ID}" --location="${REGION}" mk \
    --dataset \
    --description "Agent analytics events written by ${SERVICE_NAME}" \
    "${PROJECT_ID}:${BQAA_DATASET_ID}"
else
  echo "Dataset ${PROJECT_ID}:${BQAA_DATASET_ID} already exists; skipping."
fi

# 3. Create the runtime service account if missing.
echo
echo "--- Ensuring runtime service account exists ---"
if ! gcloud iam service-accounts describe "${SA_EMAIL}" --project="${PROJECT_ID}" >/dev/null 2>&1; then
  gcloud iam service-accounts create "${SERVICE_ACCOUNT}" \
    --project="${PROJECT_ID}" \
    --display-name "BQAA Cloud Run example runtime SA"
else
  echo "Service account ${SA_EMAIL} already exists; skipping."
fi

# 4. Grant IAM:
#    * BigQuery Data Editor on the dataset (writes agent_events).
#    * BigQuery User on the project (Storage Write API jobs).
#    * Vertex AI User on the project (Gemini via Vertex AI).
echo
echo "--- Granting IAM ---"
bq --project_id="${PROJECT_ID}" add-iam-policy-binding \
  --member="serviceAccount:${SA_EMAIL}" \
  --role="roles/bigquery.dataEditor" \
  "${PROJECT_ID}:${BQAA_DATASET_ID}"

gcloud projects add-iam-policy-binding "${PROJECT_ID}" \
  --member="serviceAccount:${SA_EMAIL}" \
  --role="roles/bigquery.user" \
  --condition=None >/dev/null

gcloud projects add-iam-policy-binding "${PROJECT_ID}" \
  --member="serviceAccount:${SA_EMAIL}" \
  --role="roles/aiplatform.user" \
  --condition=None >/dev/null

# 5. Deploy. --source uses Cloud Build to containerize from the directory
#    that holds the Dockerfile, so no separate `docker build && push`.
echo
echo "--- Deploying to Cloud Run ---"
gcloud run deploy "${SERVICE_NAME}" \
  --project="${PROJECT_ID}" \
  --region="${REGION}" \
  --source="${SCRIPT_DIR}" \
  --service-account="${SA_EMAIL}" \
  --allow-unauthenticated \
  --port=8080 \
  --memory=1Gi \
  --cpu=1 \
  --min-instances=0 \
  --max-instances=4 \
  --set-env-vars="GOOGLE_CLOUD_PROJECT=${PROJECT_ID},GOOGLE_CLOUD_LOCATION=${REGION},BQAA_DATASET_ID=${BQAA_DATASET_ID},BQAA_TABLE_ID=${BQAA_TABLE_ID},GEMINI_MODEL=${GEMINI_MODEL},AGENT_NAME=${AGENT_NAME}"

# 6. Print the invoke URL and a sample curl.
echo
URL="$(gcloud run services describe "${SERVICE_NAME}" \
  --project="${PROJECT_ID}" --region="${REGION}" --format='value(status.url)')"
echo "Deployed: ${URL}"
echo
echo "Smoke test:"
cat <<EOF
  curl -s -X POST "${URL}/chat" \\
    -H 'Content-Type: application/json' \\
    -d '{"user_id":"u1","session_id":"s1","message":"Hello"}' | jq
EOF
echo
echo "Then query BigQuery:"
cat <<EOF
  bq query --project_id="${PROJECT_ID}" --use_legacy_sql=false \\
    "SELECT event_type, agent, COUNT(*) AS n
       FROM \\\`${PROJECT_ID}.${BQAA_DATASET_ID}.${BQAA_TABLE_ID}\\\`
       WHERE timestamp >= TIMESTAMP_SUB(CURRENT_TIMESTAMP(), INTERVAL 1 HOUR)
       GROUP BY event_type, agent
       ORDER BY n DESC"
EOF
