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

data "archive_file" "source_archive" {
  type        = "tar.gz"
  source_dir  = "${path.module}/../../.."
  output_path = "${path.module}/model_garden_agent.tar.gz"
  excludes = [
    "terraform",
    ".venv",
    "__pycache__",
    "*.pyc",
    ".git"
  ]
}

resource "google_vertex_ai_reasoning_engine" "engine" {
  display_name = var.display_name
  description  = "Model Garden Agent deployed via Terraform (Dockerfile)"
  region       = var.region
  project      = var.project_id

  spec {
    # Configure identity type explicitly inside the spec block
    identity_type = var.use_agent_identity ? "AGENT_IDENTITY" : "SERVICE_ACCOUNT"

    agent_framework = "google-adk"

    source_code_spec {
      inline_source {
        source_archive = filebase64(data.archive_file.source_archive.output_path)
      }
      image_spec {
        # Triggers build from the Dockerfile inside the source archive
      }
    }

    # Configure runtime environment variables and resource limits
    deployment_spec {
      min_instances          = var.min_instances
      max_instances          = var.max_instances
      container_concurrency  = 5
      resource_limits = {
        cpu    = "2"
        memory = "4Gi"
      }

      env {
        name  = "GOOGLE_GENAI_USE_ENTERPRISE"
        value = "1"
      }
      env {
        name  = "GOOGLE_CLOUD_LOCATION"
        value = var.region
      }
      env {
        name  = "MODEL_NAME"
        value = var.model_name
      }
      env {
        name  = "MODEL_LOCATION"
        value = var.model_location
      }
      env {
        name  = "LOG_LEVEL"
        value = var.log_level
      }
      # Anthropic client not compatible with mTLS
      # https://docs.cloud.google.com/iam/docs/troubleshoot-auth-manager#401-error
      # https://github.com/anthropics/anthropic-sdk-python/blob/main/src/anthropic/lib/vertex/_client.py
      # TODO: Add support for mTLS when the Anthropic client is updated to support it.
      env {
        name  = "GOOGLE_API_PREVENT_AGENT_TOKEN_SHARING_FOR_GCP_SERVICES"
        value = "false"
      }

      # Google Cloud Observability & OpenTelemetry Instrumentation
      # Ref: https://docs.cloud.google.com/stackdriver/docs/instrumentation/ai-agent-adk
      env {
        name  = "GOOGLE_CLOUD_AGENT_ENGINE_ENABLE_TELEMETRY"
        value = var.enable_telemetry ? "true" : "false"
      }
      env {
        name  = "OTEL_SERVICE_NAME"
        value = var.display_name
      }
      env {
        name  = "OTEL_PYTHON_LOGGING_AUTO_INSTRUMENTATION_ENABLED"
        value = var.enable_telemetry ? "true" : "false"
      }
      env {
        name  = "OTEL_SEMCONV_STABILITY_OPT_IN"
        value = "gen_ai_latest_experimental"
      }
      env {
        name  = "OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT"
        value = var.enable_telemetry ? var.capture_message_content : "false"
      }
    }

    # Fall back to standard service account if Agent Identity cannot be used
    service_account = var.use_agent_identity ? null : var.custom_service_account
  }
}

# Conditionally grant Agent Platform User role if principal is resolved
resource "google_project_iam_member" "agent_identity" {
  for_each = toset([
    "roles/aiplatform.expressUser", 
    "roles/browser",
    "roles/serviceusage.serviceUsageConsumer",
    "roles/telemetry.writer"
  ])

  project = var.project_id
  role    = each.key
  member  = startswith(google_vertex_ai_reasoning_engine.engine.spec[0].effective_identity, "agents.global") ? "principal://${google_vertex_ai_reasoning_engine.engine.spec[0].effective_identity}" : "serviceAccount:${google_vertex_ai_reasoning_engine.engine.spec[0].effective_identity}"
}
