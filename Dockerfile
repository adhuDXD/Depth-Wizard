FROM python:3.12-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY depthwizard depthwizard
COPY frontend frontend
COPY scripts scripts
RUN python scripts/download_model.py
EXPOSE 8000
CMD ["python", "-m", "depthwizard.server", "--host", "0.0.0.0", "--port", "8000"]
