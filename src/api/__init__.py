"""
src/api/__init__.py
"""
from src.api.routes import health_router, predict_router

__all__ = ["predict_router", "health_router"]
