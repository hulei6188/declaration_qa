FROM python:3.11-slim

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        libreoffice-writer \
        fonts-noto-cjk \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src ./src
COPY sql ./sql

ENV PYTHONPATH=/app/src \
    GOV_SERVICE_AGENT_LEGACY_DOC_CONVERTER_COMMAND=/usr/bin/soffice

EXPOSE 8010
CMD ["uvicorn", "gov_service_agent.api.server.app:app", "--host", "0.0.0.0", "--port", "8010"]
