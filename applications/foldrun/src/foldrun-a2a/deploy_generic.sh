#!/bin/bash
PROJECT_ID=$1
SERVICE_NAME=$2
MODEL_NAME=${3:-gemini-2.5-flash}

if [ -z "$PROJECT_ID" ] || [ -z "$SERVICE_NAME" ]; then
  echo "Usage: bash deploy.sh <YOUR_PROJECT_ID> <YOUR_SERVICE_NAME> [MODEL_NAME]"
  exit 1
fi

echo "Deploying $SERVICE_NAME to project $PROJECT_ID using model $MODEL_NAME..."

gcloud run deploy $SERVICE_NAME \
  --source . \
  --region us-central1 \
  --project $PROJECT_ID \
  --set-env-vars="MODEL_NAME=$MODEL_NAME,GOOGLE_GENAI_USE_VERTEXAI=1" \
  --allow-unauthenticated \
  --format="value(status.url)" > service_url.txt

AGENT_URL=$(cat service_url.txt)
echo "Deployed to: $AGENT_URL"

# We can also update the Cloud Run service to set AGENT_URL
gcloud run services update $SERVICE_NAME \
  --region us-central1 \
  --project $PROJECT_ID \
  --set-env-vars="AGENT_URL=$AGENT_URL"

echo "Done! Agent is running at $AGENT_URL"
