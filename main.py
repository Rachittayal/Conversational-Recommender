import logging
import sys
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.responses import JSONResponse

from agent import Agent
from catalog import load_catalog
from retriever import Retriever
from schemas import (
    ChatRequest,
    ChatResponse,
    HealthResponse,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    stream=sys.stdout,
)

logger=logging.getLogger(__name__)

agent=None
startup_error=None


@asynccontextmanager
async def lifespan(app: FastAPI):

    global agent, startup_error

    try:
        logger.info("Loading catalog...")

        items=await load_catalog()
        logger.info("Loaded %d catalog items", len(items))
        logger.info("Building FAISS index...")

        retriever = Retriever()
        retriever.build_index(items)

        logger.info("Creating agent...")
        agent = Agent(retriever)
        logger.info("Server ready")

    except Exception as e:

        startup_error = str(e)

        logger.error("Startup failed: %s", e)

    yield

    logger.info("Server shutting down")


app = FastAPI(
    title="SHL Assessment Recommender",
    version="1.0.0",
    lifespan=lifespan,
)

@app.get("/health", response_model=HealthResponse)
async def health():

    if startup_error:
        return JSONResponse(status_code=503,content={"status": f"startup failed: {startup_error}"},)

    if agent is None:
        return JSONResponse(status_code=503,content={"status": "starting up"},)

    return HealthResponse(status="ok")


@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):

    if agent is None:
        return ChatResponse(
            reply="Service is starting up, please retry.",
            recommendations=None,
            end_of_conversation=False
        )

    try:
        return await agent.respond(request)

    except Exception as e:

        logger.error("Chat request failed: %s", e)

        return ChatResponse(
            reply="Something went wrong. Please try again.",
            recommendations=None,
            end_of_conversation=False,
        )