# Flask advising API (the React client and Node auth service are deployed separately)
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt gunicorn==23.0.0

COPY IAAS/LLMAcadamic-Advisor ./advisor
WORKDIR /app/advisor

RUN useradd --create-home --uid 10001 app && chown -R app /app/advisor
USER app

# OPENAI_API_KEY (and optional LANGCHAIN_* / TAVILY_API_KEY) are provided at runtime, never baked in
EXPOSE 5002
CMD ["gunicorn", "--bind", "0.0.0.0:5002", "--workers", "2", "--timeout", "300", "api.api:app"]
