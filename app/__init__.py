from flask import Flask
from core.config import Config
from flask_cors import CORS

def create_app():
    app = Flask(__name__)
    CORS(app)
    app.config.from_object(Config)

    # Import routes here to avoid circular dependencies
    from app.routes import api_bp
    app.register_blueprint(api_bp, url_prefix='/api')

    return app