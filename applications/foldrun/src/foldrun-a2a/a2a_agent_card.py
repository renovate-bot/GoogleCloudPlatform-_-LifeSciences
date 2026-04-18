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

"""FoldRun A2A Agent Card definition."""

import os
from a2a.types import AgentSkill
from vertexai.preview.reasoning_engines.templates.a2a import create_agent_card

foldrun_skills = [
    AgentSkill(
        id="protein_structure_prediction",
        name="Protein Structure Prediction",
        description=(
            "Submit AlphaFold2 (monomer/multimer) and OpenFold3 "
            "(protein + RNA/DNA/ligand) structure prediction jobs on "
            "Vertex AI Pipelines with automatic GPU selection."
        ),
        tags=["Biology", "Protein", "AlphaFold", "OpenFold3", "Structure Prediction"],
        examples=[
            "Predict the structure of ubiquitin",
            "Submit an AlphaFold2 multimer prediction for chains A and B",
            "Fold this protein with ATP using OpenFold3",
        ],
    ),
    AgentSkill(
        id="job_management",
        name="Job Management",
        description=(
            "List, monitor, and manage protein folding pipeline jobs. "
            "Check GPU quota, track progress, retry failed jobs, and "
            "delete completed jobs."
        ),
        tags=["Jobs", "Monitoring", "GPU", "Pipeline"],
        examples=[
            "List my running jobs",
            "Check the status of job 12345",
            "What GPU quota do I have available?",
        ],
    ),
    AgentSkill(
        id="results_analysis",
        name="Results Analysis & Visualization",
        description=(
            "Analyze prediction quality (pLDDT, PAE, ipTM), run parallel "
            "analysis across all models, and open interactive 3D structure "
            "viewers for both AlphaFold2 and OpenFold3 results."
        ),
        tags=["Analysis", "pLDDT", "PAE", "Visualization", "3D Viewer"],
        examples=[
            "Analyze the results from my latest job",
            "What's the pLDDT score for job 12345?",
            "Open the 3D viewer for my prediction",
        ],
    ),
    AgentSkill(
        id="alphafold_database",
        name="AlphaFold Database Queries",
        description=(
            "Query the AlphaFold Protein Structure Database for existing "
            "predictions, annotations, and variant effects before running "
            "new predictions."
        ),
        tags=["Database", "AlphaFold DB", "UniProt"],
        examples=[
            "Check if there's already a structure for P69905",
            "What annotations exist for human hemoglobin?",
        ],
    ),
]

foldrun_agent_card = create_agent_card(
    agent_name="FoldRun Agent",
    description=(
        "Expert AI assistant for protein structure prediction using "
        "AlphaFold2 and OpenFold3 on Google Cloud. Handles job submission, "
        "monitoring, quality analysis, and 3D visualization."
    ),
    skills=foldrun_skills,
    streaming=True,
)
foldrun_agent_card.preferred_transport = "JSONRPC"

agent_url = os.environ.get("AGENT_BASE_URL", "http://localhost:8080")
if agent_url.endswith("/"):
    agent_url = agent_url[:-1]
foldrun_agent_card.url = agent_url
