"""HTTP adapter: one router per concern, assembled by app.create_app()."""
from .app import create_app

__all__ = ["create_app"]
