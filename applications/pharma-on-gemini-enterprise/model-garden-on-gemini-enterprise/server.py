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

import asyncio
import inspect
import json
import logging
import os
import uvicorn
import agentplatform
from fastapi import FastAPI, HTTPException, encoders, responses
from pydantic import BaseModel
from agentplatform import agent_engines

# Import your agent instance
from model_garden_agent.agent import root_agent

# Initialize FastAPI app
app = FastAPI(
    title="Model Garden Agent Server",
    description="FastAPI wrapper for serving ADK agent on Agent Platform Agent Runtime",
)

class QueryRequest(BaseModel):
    input: dict | None = None
    class_method: str | None = None

# Initialize Vertex AI with project and location from environment variables
project_id = os.getenv("GOOGLE_CLOUD_PROJECT")
location = os.getenv("GOOGLE_CLOUD_LOCATION", "us-central1")

if project_id:
    agentplatform.init(
        project=project_id,
        location=location,
    )

# Instantiate the AdkApp template wrapping our agent.
# This manages internal services (sessions, telemetry) for the ADK agent.
adk_app = agent_engines.AdkApp(
    agent=root_agent
)

def _encode_chunk_to_json(chunk):
    """Encodes a chunk to a JSON string with a newline."""
    try:
        json_chunk = encoders.jsonable_encoder(chunk)
        return json.dumps(json_chunk) + "\n"
    except Exception:
        logging.exception("Failed to encode chunk")
        return None

async def json_generator(output):
    if hasattr(output, "__aiter__"):
        async for chunk in output:
            encoded_chunk = _encode_chunk_to_json(chunk)
            if encoded_chunk is None:
                raise RuntimeError("Failed to encode chunk in stream")
            yield encoded_chunk
    else:
        for chunk in output:
            encoded_chunk = _encode_chunk_to_json(chunk)
            if encoded_chunk is None:
                raise RuntimeError("Failed to encode chunk in stream")
            yield encoded_chunk

async def _invoke_callable_or_raise(invocation_callable, invocation_payload):
    if inspect.iscoroutinefunction(invocation_callable):
        return await invocation_callable(**invocation_payload)
    else:
        return await asyncio.to_thread(invocation_callable, **invocation_payload)

@app.post("/api/reasoning_engine")
async def query(request: QueryRequest) -> responses.JSONResponse:
    """Endpoint for standard query predictions."""
    if not hasattr(adk_app, request.class_method):
        raise HTTPException(
            status_code=400,
            detail=f"Method '{request.class_method}' not found on AdkApp",
        )
    method = getattr(adk_app, request.class_method)
    output = await _invoke_callable_or_raise(method, request.input or {})

    try:
        json_serialized_content = encoders.jsonable_encoder({"output": output})
    except ValueError as encoding_error:
        logging.exception(
            "FastAPI could not JSON-encode the response from invocation method"
            " %s. Error: %s. Invocation method's original response: %r",
            request.class_method, encoding_error, output,
        )
        raise encoding_error
    return responses.JSONResponse(content=json_serialized_content)

@app.post("/api/stream_reasoning_engine")
async def stream_query(request: QueryRequest) -> responses.StreamingResponse:
    """Endpoint for streaming query predictions."""
    if not hasattr(adk_app, request.class_method):
        raise HTTPException(
            status_code=400,
            detail=f"Method '{request.class_method}' not found on AdkApp",
        )
    method = getattr(adk_app, request.class_method)
    output = await _invoke_callable_or_raise(method, request.input or {})
    return responses.StreamingResponse(
        content=json_generator(output),
        media_type="application/json",
    )

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    uvicorn.run(app, host="0.0.0.0", port=port)
