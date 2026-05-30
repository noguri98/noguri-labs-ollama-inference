import asyncio
import json
import uuid

import redis.asyncio as redis
from fastapi import APIRouter, HTTPException

from api.ollama.crud import (
    QUEUE_NAME,
    VALKEY_HOST,
    call_ollama_chat,
    check_ollama_health,
)
from api.ollama.models import ChatRequest, ChatResponse

router = APIRouter(prefix="/ollama", tags=["ollama"])


@router.get("/health")
async def ollama_health():
    is_alive = await check_ollama_health()
    if is_alive:
        return {"status": "connected", "server": "ollama"}
    else:
        raise HTTPException(status_code=503, detail="Ollama server is unreachable")


@router.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    request_id = str(uuid.uuid4())
    r = redis.Redis(host=VALKEY_HOST, port=6379, decode_responses=True)

    # Subscribe to the response channel before sending the task
    pubsub = r.pubsub()
    await pubsub.subscribe(f"response:{request_id}")

    # Push the task to the queue
    task_data = {"request_id": request_id, "payload": request.model_dump()}
    await r.lpush(QUEUE_NAME, json.dumps(task_data))

    try:
        # Wait for the response with a timeout (e.g., 60 seconds)
        async for message in pubsub.listen():
            if message["type"] == "message":
                data = json.loads(message["data"])
                return ChatResponse(**data)
    except asyncio.TimeoutError:
        raise HTTPException(status_code=504, detail="Inference timeout")
    finally:
        await pubsub.unsubscribe(f"response:{request_id}")
        await r.close()
