import logging
import os
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
from logging import getLogger
from app.routes import data, training

logger = getLogger(__name__)
logger.setLevel(logging.DEBUG)


def create_app():
    logger.debug("Creating FastAPI application")
    # Load environment variables
    load_dotenv()

    # Get allowed origins from environment variable
    allowed_origins = os.getenv("ALLOWED_ORIGINS", "http://localhost:3000").split(",")
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

    app.include_router(data.router)
    app.include_router(training.router)

    # Root endpoint
    @app.get("/")
    async def root():
        return {"message": "This is the API for the Backend AI Trainer"}

    # Status endpoint
    @app.get("/status")
    async def status():
        return {"status": "ok", "message": "API is running"}

    return app