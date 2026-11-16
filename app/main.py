"""Application entry point: builds the FastAPI app and registers the routes."""

from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.api.dependencies import retriever
from app.api.routes import router


@asynccontextmanager
async def lifespan(_: FastAPI):
    retriever()  # Embed and index the snippets once, at startup.
    yield


app = FastAPI(title="BEON.tech Interview", lifespan=lifespan)
app.include_router(router)
