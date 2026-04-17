#!/usr/bin/env bash
# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Deploy FoldRun A2A agent to Cloud Run.
#
# Usage:
#   bash deploy_a2a_cloudrun.sh <PROJECT_ID> [SERVICE_NAME] [REGION]
#
# Example:
#   bash deploy_a2a_cloudrun.sh losiern-foldrun6
#   bash deploy_a2a_cloudrun.sh losiern-foldrun6 foldrun-a2a us-central1

set -euo pipefail

PROJECT_ID="${1:?Usage: deploy_a2a_cloudrun.sh <PROJECT_ID> [SERVICE_NAME] [REGION]}"
SERVICE_NAME="${2:-foldrun-a2a}"
REGION="${3:-us-central1}"
SERVICE_ACCOUNT="foldrun-agent-sa@${PROJECT_ID}.iam.gserviceaccount.com"

# Source .env for any env vars needed at deploy time
if [[ -f .env ]]; then
  set -a
  source .env
  set +a
fi

echo "╔═══════════════════════════════════════════════════════════╗"
echo "║   Deploying FoldRun A2A to Cloud Run                     ║"
echo "╚═══════════════════════════════════════════════════════════╝"
echo ""
echo "  Project:         ${PROJECT_ID}"
echo "  Service:         ${SERVICE_NAME}"
echo "  Region:          ${REGION}"
echo "  Service Account: ${SERVICE_ACCOUNT}"
echo ""

# Set the project
gcloud config set project "${PROJECT_ID}"

# Build env vars string for Cloud Run
# Pull key vars from .env, skip reserved/empty ones
ENV_VARS="GOOGLE_GENAI_USE_VERTEXAI=1"
ENV_VARS="${ENV_VARS},GOOGLE_CLOUD_PROJECT=${PROJECT_ID}"
ENV_VARS="${ENV_VARS},GOOGLE_CLOUD_REGION=${REGION}"

# Forward optional env vars from .env if set
for var in GCP_PROJECT_ID GCP_REGION GCS_BUCKET_NAME GEMINI_MODEL \
           OPENFOLD3_COMPONENTS_IMAGE AF2_VIEWER_URL FOLDRUN_VIEWER_URL \
           ANALYSIS_VIEWER_BASE_URL VERTEX_PROJECT_ID VERTEX_LOCATION \
           VERTEX_STAGING_BUCKET NFS_SERVER_IP NFS_SHARE_PATH; do
  val="${!var:-}"
  if [[ -n "${val}" ]]; then
    ENV_VARS="${ENV_VARS},${var}=${val}"
  fi
done

echo "Deploying to Cloud Run (source-based build)..."
echo ""

gcloud run deploy "${SERVICE_NAME}" \
  --source . \
  --region "${REGION}" \
  --platform managed \
  --service-account "${SERVICE_ACCOUNT}" \
  --set-env-vars "${ENV_VARS}" \
  --memory 2Gi \
  --cpu 2 \
  --timeout 300 \
  --min-instances 0 \
  --max-instances 5 \
  --no-allow-unauthenticated \
  --quiet

# Get the service URL
AGENT_URL=$(gcloud run services describe "${SERVICE_NAME}" \
  --region "${REGION}" \
  --format 'value(status.url)')

# Get project number for IAM setup
PROJECT_NUMBER=$(gcloud projects describe "${PROJECT_ID}" --format='value(projectNumber)')

echo ""
echo "╔═══════════════════════════════════════════════════════════╗"
echo "║   Deployment successful!                                  ║"
echo "╚═══════════════════════════════════════════════════════════╝"
echo ""
echo "  Service URL:  ${AGENT_URL}"
echo "  Agent Card:   ${AGENT_URL}/.well-known/agent.json"
echo ""
echo "┌─────────────────────────────────────────────────────────────┐"
echo "│  Next Steps                                                 │"
echo "├─────────────────────────────────────────────────────────────┤"
echo "│                                                             │"
echo "│  1. Register with Gemini Enterprise:                        │"
echo "│     bash register_a2a_gemini.sh \\                           │"
echo "│       ${PROJECT_NUMBER} \\                                   │"
echo "│       <ENGINE_ID> \\                                         │"
echo "│       ${AGENT_URL}                                          │"
echo "│                                                             │"
echo "│  2. Grant Cloud Run Invoker to Discovery Engine SA:         │"
echo "│     gcloud run services add-iam-policy-binding \\            │"
echo "│       ${SERVICE_NAME} \\                                     │"
echo "│       --region=${REGION} \\                                  │"
echo "│       --member=serviceAccount:service-${PROJECT_NUMBER}@gcp-sa-discoveryengine.iam.gserviceaccount.com \\  │"
echo "│       --role=roles/run.invoker                              │"
echo "│                                                             │"
echo "│  3. Test the agent card:                                    │"
echo "│     curl -H \"Authorization: Bearer \$(gcloud auth print-access-token)\" \\  │"
echo "│       ${AGENT_URL}/.well-known/agent.json                   │"
echo "│                                                             │"
echo "│  4. Gemini CLI setup:                                       │"
echo "│     ~/.gemini/settings.json:                                │"
echo "│       {\"experimental\": {\"enableAgents\": true}}              │"
echo "│                                                             │"
echo "│     ~/.gemini/agents/foldrun.md:                            │"
echo "│       ---                                                   │"
echo "│       kind: remote                                          │"
echo "│       name: foldrun                                         │"
echo "│       agent_card_url: ${AGENT_URL}/.well-known/agent.json   │"
echo "│       ---                                                   │"
echo "│                                                             │"
echo "└─────────────────────────────────────────────────────────────┘"
echo ""

# Write deployment metadata
cat > deployment_metadata_a2a.json <<EOF
{
  "deployment_target": "cloud_run",
  "service_name": "${SERVICE_NAME}",
  "project_id": "${PROJECT_ID}",
  "project_number": "${PROJECT_NUMBER}",
  "region": "${REGION}",
  "agent_url": "${AGENT_URL}",
  "agent_card_url": "${AGENT_URL}/.well-known/agent.json",
  "deployment_timestamp": "$(date -Iseconds)"
}
EOF

echo "Deployment metadata written to deployment_metadata_a2a.json"
