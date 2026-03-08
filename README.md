# Trademarkia AI/ML Engineer Assignment

Lightweight semantic search system with fuzzy clustering, semantic cache, and FastAPI.

## Run with Docker (recommended)
```bash
docker build -t semantic-cache-api .
docker run -p 8000:8000 semantic-cache-api
```
Open http://localhost:8000/docs

## Run with venv
```bash
python -m venv venv
source venv/Scripts/activate
pip install -r requirements.txt
uvicorn app:app --reload
```

## Endpoints
- POST /query
- GET /cache/stats  
- DELETE /cache

## Notes
- Download search_index.pkl from Colab before running
- Place search_index.pkl in the same folder as app.py
