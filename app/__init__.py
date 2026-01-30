import logging
import os
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
from logging import getLogger
from app.routes import training, models, inference, router
from paths import DATA_PATH, TRAINED_MODEL_WEIGHTS_PATH, LOG_PATH, TRAINED_MODEL_INFO_PATHS
from models.register_models import register_models
from app.state import MODEL_REGISTRY

logger = getLogger(__name__)
logger.setLevel(logging.DEBUG)


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup code
    logger.debug("Starting up the Semantic Segmentation Service")
    logger.debug("Registering models in the MODEL_REGISTRY")
    register_models(MODEL_REGISTRY)
    yield
    # Shutdown code
    logger.debug("Shutting down the Prompted Segmentation Service")


def create_app():
    logger.debug("Creating FastAPI application")
    # Load environment variables
    load_dotenv()

    # Get allowed origins from environment variable
    allowed_origins = os.getenv("ALLOWED_ORIGINS", "http://localhost:8000").split(",")
    logger.debug(f"Allowed origins: {allowed_origins}")

    app = FastAPI(
        title="Backend AI Trainer",
        lifespan=lifespan,
        description="FastAPI backend for training automatic segmentation models",
        version="0.1.0",
    )

    # Configure CORS
    app.add_middleware(
        CORSMiddleware,
        allow_origins=allowed_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    app.include_router(router)
    app.include_router(training.router)
    app.include_router(models.router)
    app.include_router(inference.router)

    os.makedirs(DATA_PATH, exist_ok=True)
    os.makedirs(TRAINED_MODEL_WEIGHTS_PATH, exist_ok=True)
    os.makedirs(LOG_PATH, exist_ok=True)
    os.makedirs(TRAINED_MODEL_INFO_PATHS, exist_ok=True)

    # Root endpoint
    @app.get("/")
    async def root():
        return {"message": "This is the API for the Backend AI Trainer"}

    # Status endpoint
    @app.get("/status")
    async def status():
        return {"status": "ok", "message": "API is running"}

    return app
