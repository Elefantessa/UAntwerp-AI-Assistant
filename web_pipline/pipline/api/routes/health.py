"""
Health Routes Blueprint
=======================

API endpoints for health checks and system info.
"""

import logging
from flask import Blueprint, jsonify

logger = logging.getLogger(__name__)


def create_health_blueprint(rag_service) -> Blueprint:
    """
    Create health blueprint with RAG service.

    Args:
        rag_service: RAGService instance

    Returns:
        Flask Blueprint
    """
    bp = Blueprint('health', __name__, url_prefix='/api')

    @bp.route("/health", methods=["GET"])
    def health():
        """Health check endpoint."""
        return jsonify(rag_service.health_check())

    @bp.route("/stats", methods=["GET"])
    def stats():
        """System statistics endpoint."""
        return jsonify(rag_service.get_statistics())

    @bp.route("/system-info", methods=["GET"])
    def system_info():
        """System configuration information."""
        return jsonify(rag_service.get_system_info())

    return bp
