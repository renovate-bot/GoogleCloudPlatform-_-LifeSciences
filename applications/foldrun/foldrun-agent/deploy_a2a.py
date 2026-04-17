#!/usr/bin/env python3
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

"""Deploy FoldRun as an A2A agent to Vertex AI Agent Engine.

Creates a separate Agent Engine instance with A2A protocol support,
leaving the existing AdkApp deployment untouched.

Usage:
    cd foldrun-agent
    python deploy_a2a.py

Requires:
    - gcloud auth application-default login
    - .env file with GCP_PROJECT_ID, GCP_REGION, etc.
"""

import json
import logging
import os
import sys
import datetime

import vertexai
from dotenv import load_dotenv, dotenv_values
from vertexai import agent_engines
from vertexai.preview.reasoning_engines import A2aAgent

# Ensure foldrun_app is importable
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

DISPLAY_NAME = "FoldRun_Agent_A2A"
DESCRIPTION = (
    "Expert AI assistant for FoldRun (AlphaFold2 + OpenFold3) protein structure "
    "prediction, job management, and results analysis — A2A protocol endpoint."
)


def load_env_vars() -> dict[str, str]:
    """Load env vars from .env, filtering reserved ones."""
    env_paths = [".env", "foldrun_app/.env", "../.env"]
    for path in env_paths:
        if os.path.exists(path):
            logger.info(f"Loading environment from {path}")
            loaded = dotenv_values(path)
            break
    else:
        logger.warning("No .env file found")
        loaded = {}

    reserved = {
        "GOOGLE_CLOUD_PROJECT",
        "GOOGLE_CLOUD_LOCATION",
        "PYTHONPATH",
        "USER",
    }
    return {k: v for k, v in loaded.items() if k not in reserved}


def find_existing_agent(display_name: str) -> str | None:
    """Check if an agent with this display name already exists."""
    try:
        for agent in agent_engines.list():
            if agent.display_name == display_name:
                return agent.resource_name
    except Exception as e:
        logger.warning(f"Could not list existing agents: {e}")
    return None


def main():
    load_dotenv()

    project = os.environ.get("GCP_PROJECT_ID")
    location = os.environ.get("GCP_REGION", "us-central1")
    service_account = f"foldrun-agent-sa@{project}.iam.gserviceaccount.com"

    if not project:
        logger.error("GCP_PROJECT_ID not set. Check your .env file.")
        sys.exit(1)

    # Load env vars for the deployed agent
    env_vars = load_env_vars()
    env_vars["GOOGLE_CLOUD_REGION"] = location
    env_vars["NUM_WORKERS"] = "1"
    env_vars["GOOGLE_CLOUD_AGENT_ENGINE_ENABLE_TELEMETRY"] = "true"
    env_vars["OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT"] = "false"

    # Build the A2A agent (imports trigger module-level vertexai.init)
    logger.info("Building A2aAgent...")
    from foldrun_app.a2a_agent_card import foldrun_agent_card
    from foldrun_app.a2a_executor import FoldRunAgentExecutor
    from foldrun_app.agent import create_alphafold_agent

    a2a_agent = A2aAgent(
        agent_card=foldrun_agent_card,
        agent_executor_builder=lambda: FoldRunAgentExecutor(
            agent=create_alphafold_agent()
        ),
    )

    # Re-initialize Vertex AI AFTER agent imports (agent.py runs
    # vertexai.init() at import time without staging_bucket, which
    # would cause agent_engines.create() to fail).
    staging_bucket = f"gs://{project}-staging"
    vertexai.init(project=project, location=location, staging_bucket=staging_bucket)
    logger.info(f"Staging bucket: {staging_bucket}")

    # Read requirements
    req_file = "foldrun_app/app_utils/.requirements.txt"
    with open(req_file) as f:
        requirements = [
            line.strip() for line in f if line.strip() and not line.startswith("#")
        ]
    logger.info(f"Loaded {len(requirements)} requirements")

    # Check for existing deployment
    existing_name = find_existing_agent(DISPLAY_NAME)

    print(f"""
    ╔═══════════════════════════════════════════════════════════╗
    ║                                                           ║
    ║   🤖 DEPLOYING FOLDRUN A2A AGENT TO AGENT ENGINE 🤖       ║
    ║                                                           ║
    ╚═══════════════════════════════════════════════════════════╝

    Project:         {project}
    Location:        {location}
    Display Name:    {DISPLAY_NAME}
    Service Account: {service_account}
    Action:          {"UPDATE" if existing_name else "CREATE"}
    """)

    logger.info("Deploying to Vertex AI Agent Engine (this can take 3-5 minutes)...")

    if existing_name:
        logger.info(f"Updating existing agent: {existing_name}")
        remote_agent = agent_engines.update(
            resource_name=existing_name,
            agent_engine=a2a_agent,
            display_name=DISPLAY_NAME,
            description=DESCRIPTION,
            requirements=requirements,
            extra_packages=["./foldrun_app"],
            env_vars=env_vars,
            service_account=service_account,
        )
    else:
        logger.info("Creating new agent...")
        remote_agent = agent_engines.create(
            agent_engine=a2a_agent,
            display_name=DISPLAY_NAME,
            description=DESCRIPTION,
            requirements=requirements,
            extra_packages=["./foldrun_app"],
            env_vars=env_vars,
            service_account=service_account,
        )

    # Extract resource info
    resource_name = remote_agent.resource_name
    parts = resource_name.split("/")
    engine_id = parts[-1]
    project_number = parts[1]

    # Build the A2A URL
    a2a_url = f"https://{location}-aiplatform.googleapis.com/v1beta1/{resource_name}/a2a"

    # Write deployment metadata
    metadata = {
        "remote_agent_engine_id": resource_name,
        "deployment_target": "agent_engine",
        "is_a2a": True,
        "a2a_url": a2a_url,
        "deployment_timestamp": datetime.datetime.now().isoformat(),
    }
    with open("deployment_metadata_a2a.json", "w") as f:
        json.dump(metadata, f, indent=2)

    # Console URL
    console_url = (
        f"https://console.cloud.google.com/vertex-ai/agents/agent-engines"
        f"/locations/{location}/agent-engines/{engine_id}"
        f"/playground?project={project}"
    )

    print(f"""
    ✅ A2A Deployment successful!

    Resource:        {resource_name}
    Service Account: {service_account}
    A2A URL:         {a2a_url}
    Console:         {console_url}

    ┌─────────────────────────────────────────────────┐
    │  Gemini CLI Setup                               │
    ├─────────────────────────────────────────────────┤
    │                                                 │
    │  1. ~/.gemini/settings.json:                    │
    │     {{"experimental": {{"enableAgents": true}}}}    │
    │                                                 │
    │  2. ~/.gemini/agents/foldrun.yaml:              │
    │     ---                                         │
    │     kind: remote                                │
    │     name: foldrun                               │
    │     agent_card_url: {a2a_url}
    │     ---                                         │
    │                                                 │
    └─────────────────────────────────────────────────┘
    """)


if __name__ == "__main__":
    main()
