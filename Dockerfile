# Multi-stage lightweight Dockerfile for Support Ticket Triage Agent
FROM python:3.11-slim AS base

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application source
COPY . .

# Initialize databases and models
RUN python -c "from app.core.database import init_db; init_db()"
RUN python -c "from app.tools.kb_search import kb_engine; kb_engine._index_articles()"
RUN python -c "from app.pipeline.classifiers import FineTunedClassifier; FineTunedClassifier()"

EXPOSE 8000 8501

ENV HOST=0.0.0.0
ENV PORT=8000
ENV STREAMLIT_PORT=8501

# Entrypoint default: launch both FastAPI backend and Streamlit dashboard
CMD ["python", "run.py", "--all"]
