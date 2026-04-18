import asyncio
import logging
import grpc
from a2a.server.request_handlers import DefaultRequestHandler, GrpcHandler
from a2a.server.tasks import InMemoryTaskStore
from a2a.grpc.a2a_pb2_grpc import add_A2AServiceServicer_to_server
from a2a.server.agent_execution import AgentExecutor

class DummyExecutor(AgentExecutor):
    async def execute(self, *args, **kwargs): pass
    async def cancel(self, *args, **kwargs): pass

async def serve():
    server = grpc.aio.server()
    request_handler = DefaultRequestHandler(
        agent_executor=DummyExecutor(),
        task_store=InMemoryTaskStore(),
    )
    grpc_handler = GrpcHandler(
        agent_card={},
        request_handler=request_handler,
    )
    add_A2AServiceServicer_to_server(grpc_handler, server)
    server.add_insecure_port('[::]:8080')
    print("Starting gRPC server on port 8080...")
    await server.start()
    await server.stop(grace=0)

if __name__ == '__main__':
    asyncio.run(serve())
