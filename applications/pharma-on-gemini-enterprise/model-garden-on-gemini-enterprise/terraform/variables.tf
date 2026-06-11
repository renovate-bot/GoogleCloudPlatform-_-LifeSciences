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

variable "project_id" {
  type        = string
  description = "The GCP Project ID"
}

variable "region" {
  type        = string
  description = "The region to deploy the Reasoning Engine"
  default     = "us-central1"
}

variable "display_name" {
  type        = string
  description = "Display name of the Reasoning Engine"
  default     = "Model Garden Agent"
}

variable "model_name" {
  type        = string
  description = "The default model name for the agent"
  default     = "claude-opus-4-7"
}

variable "model_location" {
  type        = string
  description = "The location of the model"
  default     = "global"
}

variable "log_level" {
  type        = string
  description = "Logging level for the container"
  default     = "INFO"
}

variable "enable_telemetry" {
  type        = bool
  description = "Enable OpenTelemetry and Google Cloud Observability tracing/logging"
  default     = true
}

variable "capture_message_content" {
  type        = string
  description = "Control logging of input prompts and output responses (EVENT_ONLY, true, false)"
  default     = "EVENT_ONLY"
}

variable "min_instances" {
  type        = number
  description = "Minimum container instances (set to 0 to scale-to-zero)"
  default     = 0
}

variable "max_instances" {
  type        = number
  description = "Maximum container instances"
  default     = 5
}

variable "use_agent_identity" {
  type        = bool
  description = "Use Agent Identity (requires project inside an organization)"
  default     = true
}

variable "custom_service_account" {
  type        = string
  description = "Standard service account email fallback if use_agent_identity = false"
  default     = null
}
