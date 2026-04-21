FROM python:3.10-slim
WORKDIR /app

# Install system deps (if needed) and pip dependencies
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Copy project files
COPY . .

ENV MODEL_DIR=/app/models
ENV MODEL_TYPE=auto

EXPOSE 8080
CMD ["uvicorn", "src.api.app:app", "--host", "0.0.0.0", "--port", "8080"]
