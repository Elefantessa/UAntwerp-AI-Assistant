# API Routes Module
"""Route blueprints for the Flask API."""

from .chat import create_chat_blueprint
from .health import create_health_blueprint

__all__ = ["create_chat_blueprint", "create_health_blueprint"]
