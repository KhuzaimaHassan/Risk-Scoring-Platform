"""
src/main.py
-------------
FastAPI application factory and startup/shutdown lifecycle handler.

─────────────────────────────────────────────────────────────────
Model Caching Strategy
─────────────────────────────────────────────────────────────────
The model is loaded ONCE during the lifespan context manager on startup:

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        # ── startup ──
        registry = ModelRegistry(...)
        prod_info = registry.get_latest_model()
        model, meta = load_model_with_metadata(prod_info["version"], ...)
        app.state.model         = model          # sklearn Pipeline (read-only)
        app.state.model_meta    = meta
        app.state.model_version = meta["version"]
        app.state.threshold     = meta["metrics"]["threshold_used"]
        yield
        # ── shutdown ── (cleanup if needed)

All route handlers call `request.app.state.model` — zero I/O on the hot path.

─────────────────────────────────────────────────────────────────
Application Structure
─────────────────────────────────────────────────────────────────
    /api/v1/
        POST   /predict           → score a transaction
        POST   /predict/batch     → score up to 200 transactions
        GET    /health            → liveness + readiness probe
        GET    /model-info        → current model version + metrics
        GET    /models            → all registry versions

─────────────────────────────────────────────────────────────────
Running locally:
    uvicorn src.main:app --host 0.0.0.0 --port 8000 --reload

Running in Docker:
    docker-compose up api
─────────────────────────────────────────────────────────────────
"""

from __future__ import annotations

import logging
import time
import uuid
import asyncio
from contextlib import asynccontextmanager
from pathlib import Path
from typing import AsyncGenerator

from fastapi import FastAPI, Request, status, BackgroundTasks
from fastapi.exceptions import RequestValidationError
from fastapi.encoders import jsonable_encoder
from sqlalchemy.exc import SQLAlchemyError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

from config.settings import get_settings
from src.api.routes import health_router, predict_router, auth_router, dashboard_router

logger = logging.getLogger(__name__)
settings = get_settings()


# ---------------------------------------------------------------------------
# Lifespan — model loading at startup, cleanup at shutdown
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """
    FastAPI lifespan context manager.

    STARTUP:
        1. Record start time (for uptime reporting).
        2. Discover production model version from ModelRegistry.
        3. Load sklearn Pipeline artifact from disk (joblib).
        4. Load model metadata JSON.
        5. Store everything in app.state for zero-copy access in routes.
        6. Verify DB connectivity (log warning but don't crash if DB is down).

    SHUTDOWN:
        - Currently a no-op. Add resource cleanup here (e.g. close thread pools).
    """
    # ── Startup ─────────────────────────────────────────────────────────────
    app.state.start_time = time.time()
    logger.info("=" * 60)
    logger.info("Risk Scoring Platform — API starting up …")
    logger.info("=" * 60)

    # Resolve model directory from settings or default
    model_dir = settings.model_artifact_dir
    app.state.model_dir = model_dir

    from src.services.model_state import load_bundle
    app.state.model_bundle = None
    app.state.last_retraining_attempt = None
    try:
        app.state.model_bundle = await asyncio.to_thread(load_bundle, model_dir)
    except Exception:
        logger.exception("Model load failed; readiness will fail")

    # ── Verify DB connectivity ────────────────────────────────────────────
    try:
        from sqlalchemy import text
        from src.database.session import async_db_context
        async with async_db_context() as db:
            await db.execute(text("SELECT 1"))
        logger.info("Database connection [OK]")
    except Exception as exc:
        logger.warning("Database connectivity check failed: %s — continuing startup.", exc)

    # ── Start APScheduler for automated drift detection ────────────────────
    from apscheduler.schedulers.asyncio import AsyncIOScheduler
    from src.services.drift_service import check_drift_and_retrain
    
    if settings.monitoring_enabled:
        scheduler = AsyncIOScheduler()
        scheduler.add_job(check_drift_and_retrain, "interval", minutes=5, args=[app],
                          id="drift_check", max_instances=1, coalesce=True)
        scheduler.start()
        app.state.scheduler = scheduler

    logger.info("API startup complete. Ready to serve requests.")
    logger.info("=" * 60)

    yield

    # ── Shutdown ─────────────────────────────────────────────────────────
    from src.database.session import async_engine, sync_engine
    if hasattr(app.state, "scheduler"):
        app.state.scheduler.shutdown(wait=False)
    await async_engine.dispose()
    sync_engine.dispose()
    logger.info("API shut down.")


# ---------------------------------------------------------------------------
# Application factory
# ---------------------------------------------------------------------------

def create_app() -> FastAPI:
    """
    Build and configure the FastAPI application instance.

    Separated from module level for testability — tests can call create_app()
    with overridden settings without triggering module-level side effects.
    """
    app = FastAPI(
        title="Risk Scoring Platform",
        description=(
            "Intelligent fraud detection API. Scores transactions in real-time "
            "using a versioned ML model with full prediction audit logging."
        ),
        version="1.1.0",
        docs_url="/docs",
        redoc_url="/redoc",
        openapi_url="/openapi.json",
        lifespan=lifespan,
    )

    # ── Middleware ────────────────────────────────────────────────────────
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.allowed_origins,
        allow_credentials=False,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    app.add_middleware(GZipMiddleware, minimum_size=512)

    # ── Custom exception handlers ─────────────────────────────────────────
    @app.exception_handler(RequestValidationError)
    async def validation_exception_handler(
        request: Request, exc: RequestValidationError
    ) -> JSONResponse:
        """Return a clean 422 with the error details."""
        return JSONResponse(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            content=jsonable_encoder({"error": "validation_error", "detail": exc.errors()}),
        )

    @app.exception_handler(Exception)
    async def generic_exception_handler(
        request: Request, exc: Exception
    ) -> JSONResponse:
        """Catch-all: log the exception and return a generic 500."""
        logger.exception("Unhandled exception on %s %s: %s", request.method, request.url, exc)
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={
                "error": "internal_server_error",
                "detail": "An unexpected error occurred.",
                "request_id": getattr(request.state, "request_id", None),
            },
        )

    @app.exception_handler(SQLAlchemyError)
    async def database_error(request, exc):
        logger.error("Database operation failed", exc_info=exc)
        return JSONResponse(status_code=503, content={"detail": "Database temporarily unavailable. Try again.", "request_id": getattr(request.state, "request_id", None)})

    # ── Request timing middleware ─────────────────────────────────────────
    @app.middleware("http")
    async def add_process_time_header(request: Request, call_next):
        request.state.request_id = str(uuid.uuid4())
        t0 = time.perf_counter()
        response = await call_next(request)
        elapsed_ms = int((time.perf_counter() - t0) * 1000)
        response.headers["X-Process-Time-Ms"] = str(elapsed_ms)
        response.headers["X-App-Revision"] = settings.build_revision
        response.headers["X-Request-ID"] = request.state.request_id
        return response

    @app.get("/api/v1/live", tags=["Operations"])
    async def live():
        return {"status": "ok", "revision": settings.build_revision}

    # ── Root redirect ─────────────────────────────────────────────────────
    @app.get("/", include_in_schema=False)
    async def root():
        return {
            "service": "Risk Scoring Platform",
            "version": "1.1.0",

            "docs": "/docs",
            "health": "/api/v1/health",
            "revision": settings.build_revision,
        }

    # ── Register routers ──────────────────────────────────────────────────
    API_PREFIX = "/api/v1"
    app.include_router(predict_router, prefix=API_PREFIX)
    app.include_router(health_router, prefix=API_PREFIX)
    app.include_router(auth_router, prefix=API_PREFIX)
    app.include_router(dashboard_router, prefix=API_PREFIX)

    # Serve the React Vite built frontend at /ui
    ui_dir = Path(__file__).parent.parent / "frontend" / "dist"
    if ui_dir.exists():
        app.mount("/ui", StaticFiles(directory=str(ui_dir), html=True), name="ui")

    return app


# ---------------------------------------------------------------------------
# Application singleton
# ---------------------------------------------------------------------------

app = create_app()
