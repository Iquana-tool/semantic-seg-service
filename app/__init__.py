import logging
import os
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
from logging import getLogger
from app.routes import data, training, models, segment, router
from paths import DATA_PATH, MODEL_WEIGHTS_PATH, LOG_PATH, TRAINING_RUNS_PATH
from models.register_models import register_base_models, discover_trained_models
from app.state import MODEL_REGISTRY

logger = getLogger(__name__)
logger.setLevel(logging.DEBUG)


def create_app():
    logger.debug("Creating FastAPI application")
    # Load environment variables
    load_dotenv()

    # Get allowed origins from environment variable
    allowed_origins = os.getenv("ALLOWED_ORIGINS", "http://localhost:8000").split(",")
    logger.debug(f"Allowed origins: {allowed_origins}")

    app = FastAPI(
        title="Backend AI Trainer",
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

    register_base_models(MODEL_REGISTRY)
    discover_trained_models(MODEL_REGISTRY)

    app.include_router(router)
    app.include_router(data.router)
    app.include_router(training.router)
    app.include_router(models.router)
    app.include_router(segment.router)

    os.makedirs(DATA_PATH, exist_ok=True)
    os.makedirs(MODEL_WEIGHTS_PATH, exist_ok=True)
    os.makedirs(LOG_PATH, exist_ok=True)
    os.makedirs(TRAINING_RUNS_PATH, exist_ok=True)

    # Root endpoint
    @app.get("/")
    async def root():
        return {"message": "This is the API for the Backend AI Trainer"}

    # Status endpoint
    @app.get("/status")
    async def status():
        return {"status": "ok", "message": "API is running"}

    return app
