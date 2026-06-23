FROM python:3.11-slim

WORKDIR /app

COPY backend/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY backend ./backend
COPY backend/app ./app

RUN mkdir -p /app/data

EXPOSE 5000

CMD uvicorn backend.server:app --host 0.0.0.0 --port ${PORT:-5000}
