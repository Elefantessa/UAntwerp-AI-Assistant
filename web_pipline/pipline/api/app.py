"""
Flask Application Factory
==========================

Creates and configures the Flask application with all routes.
"""

import logging
import os
from flask import Flask, render_template
from flask_cors import CORS

from config.settings import ServerConfig

logger = logging.getLogger(__name__)


def create_app(rag_service, config: ServerConfig = None) -> Flask:
    """
    Create Flask application with RAG service.

    Args:
        rag_service: RAGService instance
        config: Server configuration

    Returns:
        Configured Flask application
    """
    config = config or ServerConfig()

    # Get template directory
    template_dir = os.path.join(os.path.dirname(__file__), 'templates')

    # Create app
    app = Flask(__name__, template_folder=template_dir)

    # Configure CORS
    if config.cors_enabled:
        CORS(app, origins=config.cors_origins)

    # Register blueprints
    from .routes.chat import create_chat_blueprint
    from .routes.health import create_health_blueprint

    app.register_blueprint(create_chat_blueprint(rag_service))
    app.register_blueprint(create_health_blueprint(rag_service))

    # Index route
    @app.route("/")
    def index():
        """Main web interface."""
        try:
            return render_template("chat.html")
        except Exception as e:
            logger.error(f"Template error: {e}")
            # Fallback to inline template
            return _get_fallback_template()

    logger.info("Flask application created successfully")
    return app


def _get_fallback_template() -> str:
    """Get fallback HTML template if file not found."""
    return """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>RAG LangGraph System</title>
        <style>
            body {
                font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                max-width: 1200px; margin: 0 auto; padding: 20px;
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                min-height: 100vh;
            }
            .header {
                text-align: center; background: rgba(255,255,255,0.95);
                color: #333; padding: 30px; border-radius: 15px;
                margin-bottom: 30px;
            }
            .chat-container {
                background: rgba(255,255,255,0.95); border-radius: 15px;
                padding: 30px;
            }
            .input-group { display: flex; gap: 15px; margin-bottom: 25px; }
            #queryInput {
                flex: 1; padding: 18px; border: 2px solid #e2e8f0;
                border-radius: 10px; font-size: 16px;
            }
            #askButton {
                padding: 18px 35px; background: linear-gradient(45deg, #4f46e5, #7c3aed);
                color: white; border: none; border-radius: 10px; cursor: pointer;
            }
            .response-container {
                margin-top: 25px; padding: 25px; background: #f8fafc;
                border-radius: 12px; border-left: 5px solid #4f46e5;
            }
        </style>
    </head>
    <body>
        <div class="header">
            <h1>🧠 RAG LangGraph System</h1>
            <p>Multi-Stage Retrieval-Augmented Generation</p>
        </div>
        <div class="chat-container">
            <div class="input-group">
                <input type="text" id="queryInput" placeholder="Ask a question..." />
                <button id="askButton" onclick="askQuestion()">Ask</button>
            </div>
            <div id="responseArea"></div>
        </div>
        <script>
            function askQuestion() {
                const q = document.getElementById('queryInput').value.trim();
                if(!q) return;

                document.getElementById('responseArea').innerHTML = '<p>Processing...</p>';

                fetch('/api/query', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify({query: q})
                })
                .then(r => r.json())
                .then(data => {
                    document.getElementById('responseArea').innerHTML =
                        '<div class="response-container">' +
                        '<h4>Answer:</h4>' +
                        '<p>' + (data.answer || 'No answer') + '</p>' +
                        '<p><small>Confidence: ' + ((data.confidence || 0) * 100).toFixed(1) + '%</small></p>' +
                        '</div>';
                })
                .catch(err => {
                    document.getElementById('responseArea').innerHTML = '<p>Error: ' + err.message + '</p>';
                });
            }

            document.getElementById('queryInput').addEventListener('keypress', function(e) {
                if(e.key === 'Enter') askQuestion();
            });
        </script>
    </body>
    </html>
    """
