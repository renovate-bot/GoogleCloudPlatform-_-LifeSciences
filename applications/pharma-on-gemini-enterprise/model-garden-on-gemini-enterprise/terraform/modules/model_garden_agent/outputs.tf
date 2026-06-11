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

output "engine_id" {
  value       = google_vertex_ai_reasoning_engine.engine.id
  description = "The full resource name of the Reasoning Engine"
}

output "engine_name" {
  value       = google_vertex_ai_reasoning_engine.engine.name
  description = "The generated ID of the Reasoning Engine"
}

output "agent_identity" {
  value       = google_vertex_ai_reasoning_engine.engine.spec[0].effective_identity
  description = "The effective service account or principal URL that the reasoning engine actually runs as at runtime."
}
