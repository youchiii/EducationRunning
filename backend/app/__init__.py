"""Application package entrypoint.

This re-exports the FastAPI app instance so that modules such as
``backend.main`` can import it via ``from backend.app import app``.
"""

from .main import create_app


app = create_app()

__all__ = ["create_app", "app"]
