import os
import uvicorn
import logging

from a2a.server.apps import A2AStarletteApplication
from a2a.server.request_handlers import DefaultRequestHandler
from a2a.server.tasks import InMemoryTaskStore

from google.adk import Runner
from google.adk.a2a.executor.a2a_agent_executor import A2aAgentExecutor
from google.adk.artifacts import InMemoryArtifactService
from google.adk.memory.in_memory_memory_service import InMemoryMemoryService
from google.adk.sessions import InMemorySessionService

from foldrun_app.a2a_agent_card import foldrun_agent_card
from foldrun_app.agent import create_alphafold_agent

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def create_executor() -> A2aAgentExecutor:
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

def create_app():
    # Force Vertex AI backend for ADK
    os.environ["GOOGLE_GENAI_USE_VERTEXAI"] = "1"

    request_handler = DefaultRequestHandler(
        agent_executor=create_executor(),
        task_store=InMemoryTaskStore(),
    )

    # Initialize the Starlette Application with A2A endpoints
    server = A2AStarletteApplication(
        agent_card=foldrun_agent_card,
        http_handler=request_handler,
    )
    
    return server.build()

app = create_app()

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 8080))
    logger.info(f"Starting FoldRun A2A server on port {port}...")
    uvicorn.run("main:app", host='0.0.0.0', port=port, reload=False)
