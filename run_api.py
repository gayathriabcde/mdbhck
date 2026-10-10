from app import create_app
from core.config import Config

app = create_app()

if __name__ == '__main__':
    print(f"🚀 Starting ODP-RAG API on port {Config.PORT}...")
    app.run(host='0.0.0.0', port=Config.PORT, debug=(Config.FLASK_ENV == 'development'))