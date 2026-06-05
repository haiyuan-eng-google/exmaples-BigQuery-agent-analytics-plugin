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
#   SERVICE_NAME            (default: bqaa-cloud-run-example)
#   SERVICE_ACCOUNT         (default: bqaa-cloud-run-sa)
#   BQAA_TABLE_ID           (default: agent_events)
#   GEMINI_MODEL            (default: gemini-2.0-flash)
#   AGENT_NAME              (default: cloud_run_agent)
#   ALLOW_UNAUTHENTICATED   (default: false). Set to "true" only if you
#                           knowingly want a public endpoint that bills
#                           Vertex AI on every call.
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
ALLOW_UNAUTHENTICATED="${ALLOW_UNAUTHENTICATED:-false}"

SA_EMAIL="${SERVICE_ACCOUNT}@${PROJECT_ID}.iam.gserviceaccount.com"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo ">>> Project:              ${PROJECT_ID}"
echo ">>> Region:               ${REGION}"
echo ">>> Service:              ${SERVICE_NAME}"
echo ">>> Service account:      ${SA_EMAIL}"
echo ">>> BigQuery target:      ${PROJECT_ID}.${BQAA_DATASET_ID}.${BQAA_TABLE_ID}"
echo ">>> Model:                ${GEMINI_MODEL}"
echo ">>> Public invocation:    ${ALLOW_UNAUTHENTICATED}"
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

# 2. Create the BigQuery dataset if missing. The dataset is created in the
#    same single region as Cloud Run (REGION, e.g. us-central1). BigQuery
#    treats regions and multi-regions as distinct locations, so jobs must
#    be submitted with the matching --location -- the smoke-test query
#    below passes --location="${REGION}" for that reason.
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
#    * BigQuery Data Editor on the dataset (writes agent_events). We use
#      the dataset-ACL path (`bq update --source`) instead of
#      `bq add-iam-policy-binding --dataset` because the latter requires
#      an allowlist on some projects ("This feature requires allowlisting"
#      error). The ACL path is supported everywhere.
#    * BigQuery User on the project (Storage Write API jobs).
#    * Vertex AI User on the project (Gemini via Vertex AI).
echo
echo "--- Granting IAM ---"
_DS_JSON="$(mktemp -t bqaa-ds-XXXXXX.json)"
bq --project_id="${PROJECT_ID}" show --format=prettyjson \
  "${PROJECT_ID}:${BQAA_DATASET_ID}" > "${_DS_JSON}"
python3 - "${_DS_JSON}" "${SA_EMAIL}" <<'PY'
import json, sys
path, sa = sys.argv[1], sys.argv[2]
with open(path) as f:
    ds = json.load(f)
entry = {"role": "roles/bigquery.dataEditor", "userByEmail": sa}
access = ds.get("access", [])
if entry not in access:
    access.append(entry)
ds["access"] = access
with open(path, "w") as f:
    json.dump(ds, f)
PY
bq update --source "${_DS_JSON}" "${PROJECT_ID}:${BQAA_DATASET_ID}"
rm -f "${_DS_JSON}"

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
#    The endpoint defaults to authenticated-only; flip ALLOW_UNAUTHENTICATED
#    to "true" knowingly if you want a public endpoint.
if [[ "${ALLOW_UNAUTHENTICATED}" == "true" ]]; then
  AUTH_FLAG="--allow-unauthenticated"
  echo
  echo "WARNING: deploying with --allow-unauthenticated. The endpoint will be"
  echo "         callable by anyone on the internet and every call bills"
  echo "         Vertex AI / Gemini against ${PROJECT_ID}."
else
  AUTH_FLAG="--no-allow-unauthenticated"
fi

echo
echo "--- Deploying to Cloud Run ---"
gcloud run deploy "${SERVICE_NAME}" \
  --project="${PROJECT_ID}" \
  --region="${REGION}" \
  --source="${SCRIPT_DIR}" \
  --service-account="${SA_EMAIL}" \
  ${AUTH_FLAG} \
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

if [[ "${ALLOW_UNAUTHENTICATED}" == "true" ]]; then
  cat <<EOF
Smoke test (public endpoint):
  curl -s -X POST "${URL}/chat" \\
    -H 'Content-Type: application/json' \\
    -d '{"user_id":"u1","session_id":"s1","message":"Hello"}' | jq
EOF
else
  cat <<EOF
Smoke test (authenticated endpoint -- attach a Google ID token):
  TOKEN="\$(gcloud auth print-identity-token)"
  curl -s -X POST "${URL}/chat" \\
    -H "Authorization: Bearer \${TOKEN}" \\
    -H 'Content-Type: application/json' \\
    -d '{"user_id":"u1","session_id":"s1","message":"Hello"}' | jq

  Anyone who needs to call this service must be granted the
  roles/run.invoker role on it:
    gcloud run services add-iam-policy-binding "${SERVICE_NAME}" \\
      --project="${PROJECT_ID}" --region="${REGION}" \\
      --member="user:you@example.com" --role="roles/run.invoker"
EOF
fi

echo
echo "Then query BigQuery:"
cat <<EOF
  bq query --project_id="${PROJECT_ID}" --location="${REGION}" \\
    --use_legacy_sql=false \\
    "SELECT event_type, agent, COUNT(*) AS n
       FROM \\\`${PROJECT_ID}.${BQAA_DATASET_ID}.${BQAA_TABLE_ID}\\\`
       WHERE timestamp >= TIMESTAMP_SUB(CURRENT_TIMESTAMP(), INTERVAL 1 HOUR)
       GROUP BY event_type, agent
       ORDER BY n DESC"
EOF
