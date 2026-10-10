# To Run the Application

## 1. Install the Requirements

Install the requirements using:

```pip install -r requirements.txt```


## 2. Create a `.env` File

Create a `.env` file in the project root folder.

``` 
    MONGODBURI= DATABASENAME=odp_rag

    FLASKAPP=runapi.py FLASK_ENV=development PORT=5000
    Redis Settings (Message Broker for Celery)

    REDIS_URI=redis://localhost:6379/0
```


## 3. Running the System (3 Terminals Required)

### Terminal 1: Start Redis Server

```redis-server```


### Terminal 2: Start the Celery Background Worker

Make sure your virtual environment is active.

To activate the virtual environment:

    **Create a virtual environment:**

    python -m venv venv


    **Linux/macOS:**

    source venv/bin/activate


    **Windows:**

    venv\Scripts\activate`


Start the Celery worker:

```celery -A worker.tasks.celery_app worker --loglevel=info```


### Terminal 3: Start the FastAPI Application Server

```python run_api.py```


## 4. Accessing & Testing the App

**Web UI:**

Open your browser and navigate to [http://localhost:8000/](http://localhost:8000/).

**Interactive API Docs (Swagger):**

Navigate to [http://localhost:8000/docs](http://localhost:8000/docs) to test endpoints like `/ingest`, `/search`, and `/analytics` directly.

## 5. Running Evaluation & Test Scripts

To test query processing or model metrics offline, run the scripts in `testing/`.

**Test a sample query against retrieval routines:**

```python -m testing.test_query```


**Evaluate baseline retrieval recall and grounding overlap:**

```python -m testing.evaluate_model```