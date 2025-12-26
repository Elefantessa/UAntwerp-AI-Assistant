"""
Chat Routes Blueprint
=====================

API endpoints for query processing.
"""

import logging
from dataclasses import asdict
from flask import Blueprint, request, jsonify

logger = logging.getLogger(__name__)


def create_chat_blueprint(rag_service) -> Blueprint:
    """
    Create chat blueprint with RAG service.

    Args:
        rag_service: RAGService instance

    Returns:
        Flask Blueprint
    """
    bp = Blueprint('chat', __name__, url_prefix='/api')

    @bp.route("/query", methods=["POST"])
    def query():
        """Process query through RAG system."""
        try:
            data = request.get_json(force=True)
            query_text = (data or {}).get("query", "").strip()

            if not query_text:
                return jsonify({"error": "Query is required"}), 400

            # Process through RAG system
            response = rag_service.process_query(query_text)

            # Convert to JSON-serializable format
            return jsonify(response.to_dict())

        except Exception as e:
            logger.exception("API query error")
            return jsonify({"error": str(e)}), 500

    return bp
