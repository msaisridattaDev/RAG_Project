"""Application entry point — exposes the ASGI app for uvicorn / gunicorn.

Usage:
    uvicorn autogen.main:app --reload
"""

from autogen.api.app import create_app

app = create_app()
