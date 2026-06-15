FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY declaration_qa ./declaration_qa
COPY sql ./sql

EXPOSE 8010
CMD ["uvicorn", "declaration_qa.server:app", "--host", "0.0.0.0", "--port", "8010"]
