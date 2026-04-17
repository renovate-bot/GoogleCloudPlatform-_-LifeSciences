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

"""FoldRun A2A Agent Engine entrypoint with direct method overrides."""

import logging

from vertexai.preview.reasoning_engines import A2aAgent

from google.adk import Runner
from google.adk.a2a.executor.a2a_agent_executor import A2aAgentExecutor
from google.adk.artifacts import InMemoryArtifactService
from google.adk.memory.in_memory_memory_service import InMemoryMemoryService
from google.adk.sessions import InMemorySessionService

from foldrun_app.a2a_agent_card import foldrun_agent_card
from foldrun_app.agent import create_alphafold_agent

logger = logging.getLogger(__name__)

def _build_executor() -> A2aAgentExecutor:
    """Build the FoldRun A2A executor with the ADK agent."""
    agent = create_alphafold_agent()
    runner = Runner(
        app_name=agent.name,
        agent=agent,
        artifact_service=InMemoryArtifactService(),
        session_service=InMemorySessionService(),
        memory_service=InMemoryMemoryService(),
    )
    return A2aAgentExecutor(runner=runner)

a2a_agent = A2aAgent(
    agent_card=foldrun_agent_card,
    agent_executor_builder=_build_executor,
)
