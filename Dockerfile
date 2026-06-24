FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src ./src
COPY sql ./sql

ENV PYTHONPATH=/app/src

EXPOSE 8010
CMD ["uvicorn", "declaration_qa.api.server.app:app", "--host", "0.0.0.0", "--port", "8010"]
