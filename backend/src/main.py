import asyncio

import uvicorn
from api.health.views import router as health_router
from fastapi import FastAPI

from api.ollama.crud import worker
from api.ollama.views import router as ollama_router

app = FastAPI(title="Labs Ollama Inference API")


@app.on_event("startup")
async def startup_event():
    # Start the worker as a background task
    asyncio.create_task(worker())


# Include routers

app.include_router(health_router)
app.include_router(ollama_router)


def main():
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)


if __name__ == "__main__":
    main()
