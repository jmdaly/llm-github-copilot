FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

# Install system dependencies for building (if needed)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Pre-copy only files needed for dependency resolution
COPY pyproject.toml ./

# Install Python dependencies
RUN pip install --upgrade pip \
    && pip install flake8 pytest \
    && pip install .

# Now copy the rest of the source code
COPY README.md llm_github_copilot.py run-tests.sh ./
COPY tests/ ./tests/

ENV PYTHONPATH=/app

CMD ["/app/run-tests.sh"]

