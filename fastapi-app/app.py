"""FastAPI application entry point."""

from fastapi import FastAPI
import uvicorn

from api import inference_router, root_router
from core import config


# Initialize FastAPI app
app = FastAPI(title=config.API_TITLE)

# Include the router
app.include_router(root_router)
app.include_router(inference_router)


if __name__ == "__main__":
    uvicorn.run("app:app", host=config.API_HOST, port=config.API_PORT, reload=True)
