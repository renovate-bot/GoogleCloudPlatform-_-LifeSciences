# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Stage 1: Build stage
FROM python:3.13-slim AS builder

COPY --from=ghcr.io/astral-sh/uv:0.11.16 /uv /uvx /bin/

ENV UV_COMPILE_BYTECODE=1

WORKDIR /app

# Copy only dependency files first to leverage caching
COPY pyproject.toml ./

RUN uv sync --no-dev --no-install-project

# Copy the rest of the application code
COPY . .

# Sync the project (installs model_garden_agent package)
RUN uv sync --no-dev --no-editable

# Stage 2: Runtime stage
FROM python:3.13-slim AS runtime

# Create a non-root user for security
RUN useradd --create-home appuser
WORKDIR /app

# Copy the virtual environment and set ownership
COPY --from=builder --chown=appuser:appuser /app/.venv /app/.venv
# Copy the application code and set ownership
COPY --chown=appuser:appuser server.py .

# Switch to non-root user
USER appuser

# Ensure we use the virtual environment
ENV PATH="/app/.venv/bin:$PATH"
ENV PYTHONUNBUFFERED=1

EXPOSE 8080

# Run the FastAPI server using uvicorn from the venv
CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8080"]
