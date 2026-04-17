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

# Register or unregister the FoldRun A2A agent with Gemini Enterprise.
#
# Usage:
#   Register:
#     bash register_a2a_gemini.sh <PROJECT_NUMBER> <ENGINE_ID> <AGENT_URL> [LOCATION]
#
#   Unregister:
#     bash register_a2a_gemini.sh --delete <PROJECT_NUMBER> <ENGINE_ID> <AGENT_ID> [LOCATION]
#
# Prerequisites:
#   - Project must be allowlisted for Gemini Enterprise A2A integration
#   - gcloud auth login

set -euo pipefail

# --- Handle --delete mode ---
if [[ "${1:-}" == "--delete" ]]; then
  shift
  PROJECT_NUMBER="${1:?Usage: register_a2a_gemini.sh --delete <PROJECT_NUMBER> <ENGINE_ID> <AGENT_ID> [LOCATION]}"
  ENGINE_ID="${2:?}"
  AGENT_ID="${3:?}"
  LOCATION="${4:-global}"

  echo "Unregistering agent ${AGENT_ID}..."
  curl -s -X DELETE \
    -H "Authorization: Bearer $(gcloud auth print-access-token)" \
    -H "Content-Type: application/json" \
    "https://discoveryengine.googleapis.com/v1alpha/projects/${PROJECT_NUMBER}/locations/${LOCATION}/collections/default_collection/engines/${ENGINE_ID}/assistants/default_assistant/agents/${AGENT_ID}"

  echo ""
  echo "Agent ${AGENT_ID} unregistered."
  exit 0
fi

# --- Register mode ---
PROJECT_NUMBER="${1:?Usage: register_a2a_gemini.sh <PROJECT_NUMBER> <ENGINE_ID> <AGENT_URL> [LOCATION]}"
ENGINE_ID="${2:?}"
AGENT_URL="${3:?}"
LOCATION="${4:-global}"

AGENT_NAME="foldrun"
AGENT_DISPLAY_NAME="FoldRun Agent"
AGENT_DESCRIPTION="Expert AI assistant for protein structure prediction using AlphaFold2 and OpenFold3 on Google Cloud. Handles job submission, monitoring, quality analysis, and 3D visualization."
PROVIDER_ORG="Google Cloud HCLS"

# Build the agent card JSON (escaped for embedding)
AGENT_CARD=$(cat <<CARD
{
  "provider": {
    "organization": "${PROVIDER_ORG}",
    "url": "${AGENT_URL}"
  },
  "name": "${AGENT_NAME}",
  "description": "${AGENT_DESCRIPTION}",
  "capabilities": {},
  "defaultInputModes": ["text/plain"],
  "defaultOutputModes": ["text/plain"],
  "skills": [
    {
      "id": "protein_structure_prediction",
      "name": "Protein Structure Prediction",
      "description": "Submit AlphaFold2 and OpenFold3 structure prediction jobs on Vertex AI Pipelines with automatic GPU selection.",
      "examples": [
        "Predict the structure of ubiquitin",
        "Submit an AlphaFold2 multimer prediction for chains A and B",
        "Fold this protein with ATP using OpenFold3"
      ],
      "tags": ["Biology", "Protein", "AlphaFold", "OpenFold3"]
    },
    {
      "id": "job_management",
      "name": "Job Management",
      "description": "List, monitor, and manage protein folding pipeline jobs. Check GPU quota, track progress, retry failed jobs.",
      "examples": [
        "List my running jobs",
        "Check the status of job 12345",
        "What GPU quota do I have available?"
      ],
      "tags": ["Jobs", "Monitoring", "GPU", "Pipeline"]
    },
    {
      "id": "results_analysis",
      "name": "Results Analysis",
      "description": "Analyze prediction quality (pLDDT, PAE, ipTM), run parallel analysis, and open interactive 3D structure viewers.",
      "examples": [
        "Analyze the results from my latest job",
        "Open the 3D viewer for my prediction"
      ],
      "tags": ["Analysis", "pLDDT", "Visualization"]
    }
  ],
  "version": "1.0.0"
}
CARD
)

# Escape the agent card JSON for embedding in the request body
ESCAPED_CARD=$(echo "${AGENT_CARD}" | python3 -c "import sys, json; print(json.dumps(sys.stdin.read().strip()))")

echo "Registering FoldRun A2A agent with Gemini Enterprise..."
echo ""
echo "  Project Number: ${PROJECT_NUMBER}"
echo "  Engine ID:      ${ENGINE_ID}"
echo "  Agent URL:      ${AGENT_URL}"
echo "  Location:       ${LOCATION}"
echo ""

RESPONSE=$(curl -s -X POST \
  -H "Authorization: Bearer $(gcloud auth print-access-token)" \
  -H "Content-Type: application/json" \
  "https://discoveryengine.googleapis.com/v1alpha/projects/${PROJECT_NUMBER}/locations/${LOCATION}/collections/default_collection/engines/${ENGINE_ID}/assistants/default_assistant/agents" \
  -d "{
  \"name\": \"${AGENT_NAME}\",
  \"displayName\": \"${AGENT_DISPLAY_NAME}\",
  \"description\": \"${AGENT_DESCRIPTION}\",
  \"a2aAgentDefinition\": {
    \"jsonAgentCard\": ${ESCAPED_CARD}
  }
}")

echo "Response:"
echo "${RESPONSE}" | python3 -m json.tool 2>/dev/null || echo "${RESPONSE}"
echo ""

# Extract agent ID if present
AGENT_ID=$(echo "${RESPONSE}" | python3 -c "import sys, json; print(json.loads(sys.stdin.read()).get('name','').split('/')[-1])" 2>/dev/null || true)

if [[ -n "${AGENT_ID}" && "${AGENT_ID}" != "" ]]; then
  echo "Agent registered successfully!"
  echo ""
  echo "  Agent ID: ${AGENT_ID}"
  echo ""
  echo "  To unregister:"
  echo "    bash register_a2a_gemini.sh --delete ${PROJECT_NUMBER} ${ENGINE_ID} ${AGENT_ID} ${LOCATION}"
  echo ""
  echo "  Don't forget to grant Cloud Run Invoker to Discovery Engine SA:"
  echo "    gcloud run services add-iam-policy-binding foldrun-a2a \\"
  echo "      --region=us-central1 \\"
  echo "      --member=serviceAccount:service-${PROJECT_NUMBER}@gcp-sa-discoveryengine.iam.gserviceaccount.com \\"
  echo "      --role=roles/run.invoker"
fi
