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

"""FoldRun A2A Agent Engine entrypoint.

Wraps the FoldRun ADK agent as an A2aAgent for deployment to
Vertex AI Agent Engine with A2A protocol support.

Deploy with:
    entrypoint_module: foldrun_app.a2a_app
    entrypoint_object: a2a_agent

Gemini CLI config (~/.gemini/agents/foldrun.yaml):
    ---
    kind: remote
    name: foldrun
    agent_card_url: https://<location>-aiplatform.googleapis.com/v1beta1/projects/<project>/locations/<location>/reasoningEngines/<id>/a2a
    ---
"""

from vertexai.preview.reasoning_engines import A2aAgent

from foldrun_app.a2a_agent_card import foldrun_agent_card
from foldrun_app.a2a_executor import FoldRunAgentExecutor
from foldrun_app.agent import create_alphafold_agent


def _build_executor() -> FoldRunAgentExecutor:
    """Build the FoldRun A2A executor with the ADK agent."""
    agent = create_alphafold_agent()
    return FoldRunAgentExecutor(agent=agent)


a2a_agent = A2aAgent(
    agent_card=foldrun_agent_card,
    agent_executor_builder=_build_executor,
)
