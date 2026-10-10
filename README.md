#1. create a .env in the project root folder

```env
FLASK_APP=run_api.py
FLASK_ENV=development
PORT=5000


# Redis Settings (Message Broker for Celery)
# If running Redis locally, this is the default URI
REDIS_URI=redis://localhost:6379/0
```
