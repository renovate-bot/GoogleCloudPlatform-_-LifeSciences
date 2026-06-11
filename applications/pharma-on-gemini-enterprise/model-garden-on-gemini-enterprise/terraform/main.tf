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

terraform {
  required_version = ">= 1.0"
  required_providers {
    google = {
      source  = "hashicorp/google"
      version = ">= 7.28.0" # Required for image_spec block support
    }
    archive = {
      source  = "hashicorp/archive"
      version = ">= 2.0.0"
    }
  }
}

provider "google" {
  project = var.project_id
  region  = var.region
}

module "model_garden_agent" {
  source                  = "./modules/model_garden_agent"
  project_id              = var.project_id
  region                  = var.region
  display_name            = var.display_name
  model_name              = var.model_name
  model_location          = var.model_location
  log_level               = var.log_level
  enable_telemetry        = var.enable_telemetry
  capture_message_content = var.capture_message_content
  min_instances           = var.min_instances
  max_instances           = var.max_instances
  use_agent_identity      = var.use_agent_identity
  custom_service_account  = var.custom_service_account
}
