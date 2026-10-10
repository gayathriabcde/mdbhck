import os

from flask import Flask, send_from_directory
from flask_cors import CORS

from core.config import Config

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def create_app():
    app = Flask(__name__)
    CORS(app)
    app.config.from_object(Config)

    @app.route("/")
    def index_page():
        return send_from_directory(PROJECT_ROOT, "index.html")

    # Import routes here to avoid circular dependencies
    from app.routes import api_bp
    app.register_blueprint(api_bp, url_prefix='/api')

    return app