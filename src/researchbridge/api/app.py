"""FastAPI app for the corpus explorer.

Local-only by design (blueprint Sec 44: the MVP runs on local/free
infrastructure), so CORS is opened just to the Next.js dev server rather
than to the world.
"""

from __future__ import annotations

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from researchbridge.api.benchmark_routes import router as benchmark_router
from researchbridge.api.routes import router
from researchbridge.config import load_config

DEV_FRONTEND_ORIGINS = ["http://localhost:3000", "http://127.0.0.1:3000"]


def create_app() -> FastAPI:
    load_config()

    app = FastAPI(
        title="ResearchBridge Corpus Explorer",
        description="Browse and semantically search the ingested paper corpus.",
        version="0.1.0",
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=DEV_FRONTEND_ORIGINS,
        allow_methods=["GET", "PUT"],  # PUT: the annotation workbench saves back to the YAML files
        allow_headers=["*"],
    )
    app.include_router(router)
    app.include_router(benchmark_router)

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    return app


app = create_app()
