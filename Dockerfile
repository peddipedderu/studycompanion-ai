FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc g++ libffi-dev \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY backend/ ./backend/
COPY frontend/ ./frontend/
COPY docs/ ./docs/
COPY evaluation/ ./evaluation/

RUN mkdir -p /app/backend/uploads /app/backend/chroma_db

WORKDIR /app/backend

EXPOSE 8080

CMD ["python", "main.py"]
