from fastapi import APIRouter

# Initialize the router
root_router = APIRouter(prefix="/inference", tags=["Prediction", "Inference"])


@root_router.get("/", tags=["Home Page"])
def root():
    """Default page."""
    return {"message": "Welcome to the ML Flow Service!"}
