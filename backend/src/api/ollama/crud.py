import asyncio
import json
import os

import httpx
import redis.asyncio as redis

from api.ollama.models import ChatRequest, ChatResponse

OLLAMA_SERVER = os.getenv("OLLAMA_SERVER", "localhost")
VALKEY_HOST = os.getenv("VALKEY_HOST", "localhost")
QUEUE_NAME = "ollama_tasks"


async def get_redis_client():
    return redis.Redis(host=VALKEY_HOST, port=6379, decode_responses=True)


async def call_ollama_chat(request: ChatRequest) -> ChatResponse:
    async with httpx.AsyncClient() as client:
        response = await client.post(
            f"http://{OLLAMA_SERVER}:11434/api/chat",
            json=request.model_dump(),
            timeout=None,
        )
        response.raise_for_status()
        return ChatResponse(**response.json())


async def check_ollama_health() -> bool:
    async with httpx.AsyncClient() as client:
        try:
            response = await client.get(f"http://{OLLAMA_SERVER}:11434/", timeout=5.0)
            return response.status_code == 200
        except Exception:
            return False


async def worker():
    r = await get_redis_client()
    print(f"Worker started, listening on {QUEUE_NAME}...")
    while True:
        try:
            # Wait for a task from the queue
            task = await r.brpop(QUEUE_NAME, timeout=1)
            if task:
                _, message_json = task
                data = json.loads(message_json)
                request_id = data.get("request_id")
                payload = data.get("payload")

                request = ChatRequest(**payload)
                print(f"Processing task {request_id}...")

                # Call Ollama
                result = await call_ollama_chat(request)

                # Publish the result back to a unique channel for the request
                await r.publish(f"response:{request_id}", result.model_dump_json())
                print(f"Task {request_id} completed.")
        except Exception as e:
            print(f"Worker error: {e}")
            await asyncio.sleep(1)
