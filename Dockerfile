FROM python:3.12

WORKDIR /app

COPY . . 

RUN pip install -r requirements.txt --no-cache-dir

# Bake the FlashRank reranker model into the image so Cloud Run never downloads it at runtime
ENV FLASHRANK_CACHE_DIR=/app/.flashrank
RUN python -c "from flashrank import Ranker; Ranker(cache_dir='/app/.flashrank')"


CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8080"]