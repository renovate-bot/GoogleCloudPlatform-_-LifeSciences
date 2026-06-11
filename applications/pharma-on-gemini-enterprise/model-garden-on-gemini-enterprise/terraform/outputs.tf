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

output "reasoning_engine_id" {
  value       = module.model_garden_agent.engine_id
  description = "The full resource name of the deployed Reasoning Engine"
}

output "agent_identity" {
  value       = module.model_garden_agent.agent_identity
  description = "The effective identity principal used at runtime (Agent Identity principal or standard service account email)"
}
