# One image, two roles (MCP server or agent API), chosen by the compose command.
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1
WORKDIR /app

COPY pyproject.toml ./
COPY src ./src
RUN pip install .

# Run as a non-root user.
RUN useradd --create-home app && mkdir -p /app/traces /docs && chown -R app /app
USER app

# DOCS_DIR unset -> the bundled sample docs. Mount your own folder at /docs and set DOCS_DIR=/docs.
ENV TRACE_DIR=/app/traces

EXPOSE 8000 8001
CMD ["uvicorn", "introspective_agent.api:app", "--host", "0.0.0.0", "--port", "8000"]
