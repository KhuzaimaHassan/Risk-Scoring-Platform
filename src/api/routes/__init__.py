"""
src/api/routes/__init__.py
"""
from src.api.routes.predict import router as predict_router
from src.api.routes.health import router as health_router

__all__ = ["predict_router", "health_router"]
