"""WSGI entrypoint for Gunicorn / Docker."""

from app.main import app
from app.model import load_model

load_model()
