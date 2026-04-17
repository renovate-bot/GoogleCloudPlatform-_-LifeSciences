import asyncio
import os
import httpx
from uuid import uuid4
from typing import Any

from a2a.client import ClientFactory
from a2a.client.client import ClientConfig
from a2a.client.middleware import ClientCallInterceptor, ClientCallContext
from a2a.types import Message, TextPart, Role, TransportProtocol, AgentCard

class SimpleAuthInterceptor(ClientCallInterceptor):
    def __init__(self, token: str):
        self.token = token
        
    async def intercept(
        self,
        method_name: str,
        request_payload: dict[str, Any],
        http_kwargs: dict[str, Any],
        agent_card: AgentCard | None,
        context: ClientCallContext | None,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        headers = http_kwargs.get('headers', {})
        headers['Authorization'] = f'Bearer {self.token}'
        http_kwargs['headers'] = headers
        return request_payload, http_kwargs

async def main():
    token = os.popen('gcloud auth print-access-token').read().strip()
    base_url = "https://us-central1-aiplatform.googleapis.com"
    card_path = "/v1beta1/projects/673254409461/locations/us-central1/reasoningEngines/7986545150664900608/a2a/v1/card"
    
    # Enable debug logging
    import logging
    logging.basicConfig(level=logging.DEBUG)
    
    # Connect client
    client = await ClientFactory.connect(
        agent=base_url,
        client_config=ClientConfig(
            streaming=True,
            supported_transports=[TransportProtocol.http_json],
            httpx_client=httpx.AsyncClient(headers={"Authorization": f"Bearer {token}"})
        ),
        relative_card_path=card_path,
        resolver_http_kwargs={"headers": {"Authorization": f"Bearer {token}"}}
    )
    
    print("Sending request...")
    try:
        parts = [TextPart(text='List my recent protein folding jobs.')]
        message = Message(
            role=Role.user,
            parts=parts,
            message_id=uuid4().hex,
        )

        streaming_response = client.send_message(message)

        async for chunk in streaming_response:
            task, _ = chunk
            print("EVENT TASK:", task)
            
    except Exception as e:
        print("ERROR:", e)
        
    await client.close()

asyncio.run(main())
