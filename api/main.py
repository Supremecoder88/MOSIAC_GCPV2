
import asyncpg

from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from api.dependencies import get_hitl_gate, get_procedural_store
from api.routers import analysis, signals, review, memory
from config.logging_config import setup_logging
from config.settings import settings
from config.shared_pool import (
    register_pgvector_codec,
    set_shared_pool,
)


logger = setup_logging(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application startup and shutdown."""

    logger.info("MOSAIC API starting up...")

    app.state.db_pool = None
    set_shared_pool(None)

    try:
        app.state.db_pool = await asyncpg.create_pool(
            host=settings.db_host,
            port=int(settings.db_port),
            database=settings.db_name,
            user=settings.db_user,
            password=settings.db_password,
            min_size=1,
            max_size=3,
            command_timeout=30,
            init=register_pgvector_codec,
        )

        set_shared_pool(app.state.db_pool)

        logger.info("Shared database connection pool initialized")

    except Exception as e:
        logger.error(
            f"Database pool initialization failed | error={e}"
        )

    if app.state.db_pool is not None:
        try:
            procedural_store = get_procedural_store()
            await procedural_store.initialise_defaults()

            logger.info("Default agent procedures initialised")

        except Exception as e:
            logger.error(
                f"Failed to initialise procedures | error={e}"
            )
    else:
        logger.warning(
            "Skipping default procedure initialization because "
            "the shared database pool is unavailable."
        )

    logger.info("MOSAIC API ready | version=0.1.0")

    yield

    logger.info("MOSAIC API shutting down...")

    try:
        procedural_store = get_procedural_store()
        await procedural_store.close()
        logger.info("ProceduralStore closed")

    except Exception as e:
        logger.error(
            f"Error closing ProceduralStore | error={e}"
        )

    try:
        hitl = get_hitl_gate()
        await hitl.close()
        logger.info("HITLGate closed")

    except Exception as e:
        logger.error(
            f"Error closing HITLGate | error={e}"
        )

    if app.state.db_pool is not None:
        try:
            await app.state.db_pool.close()
            logger.info("Shared database pool closed")

        except Exception as e:
            logger.error(
                f"Error closing database pool | error={e}"
            )

    set_shared_pool(None)

    logger.info("MOSAIC API shutdown complete")


app = FastAPI(
    title="MOSAIC — Clinical Trial Intelligence API",
    description=(
        "Multi-Agent Operating System for AI Cognition. "
        "Detects research integrity signals across clinical trials "
        "using 6 specialist AI agents running in parallel on GCP."
    ),
    version="0.1.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)


app.add_middleware(
    CORSMiddleware,
    allow_origins=(
        ["*"] if settings.api_env == "development" else []
    ),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


app.include_router(analysis.router, prefix="/api/v1")
app.include_router(signals.router, prefix="/api/v1")
app.include_router(review.router, prefix="/api/v1")
app.include_router(memory.router, prefix="/api/v1")


@app.get(
    "/api/v1/health",
    summary="System health check",
    description=(
        "Returns system status including database connectivity "
        "and queue depth."
    ),
    tags=["System"],
)
async def health_check():
    """Return system and database health status."""

    pool = app.state.db_pool

    if pool is None:
        raise HTTPException(
            status_code=503,
            detail="Database connection pool is unavailable.",
        )

    try:
        async with pool.acquire() as conn:
            signals_count = (
                await conn.fetchval(
                    "SELECT COUNT(*) FROM signals"
                ) or 0
            )

            pending_count = (
                await conn.fetchval(
                    "SELECT COUNT(*) FROM hitl_reviews "
                    "WHERE decision = 'pending'"
                ) or 0
            )

            episodes_count = (
                await conn.fetchval(
                    "SELECT COUNT(*) FROM episodes"
                ) or 0
            )

        return {
            "status": "healthy",
            "app": "MOSAIC",
            "version": "0.1.0",
            "database": "connected",
            "details": {
                "signals_in_db": signals_count,
                "pending_reviews": pending_count,
                "episodes_count": episodes_count,
                "queue_depth": pending_count,
            },
        }

    except Exception as e:
        logger.error(
            f"Health check DB error | error={e}"
        )

        raise HTTPException(
            status_code=503,
            detail="Database health check failed.",
        )


@app.get("/", tags=["System"])
async def root():
    """Confirm that the API is running."""

    return {
        "message": "MOSAIC Clinical Trial Intelligence API",
        "version": "0.1.0",
        "docs": "/docs",
        "health": "/api/v1/health",
    }
