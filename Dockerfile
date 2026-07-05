FROM python:3.11-slim

WORKDIR /app

COPY pyproject.toml README.md ./
COPY src/ src/
COPY ui/ ui/
COPY scripts/ scripts/

RUN pip install --no-cache-dir -e .

EXPOSE 8000 8501

CMD ["uvicorn", "findociq.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
