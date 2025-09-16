FROM python:3.11-slim

WORKDIR /app

ARG DBURL
ARG DBUSERNAME
ARG DBPASSWORD

# Install system dependencies needed by OpenCV
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

RUN chmod +x /app/generate_env.sh \
    && /app/generate_env.sh

EXPOSE 5000

# Run with gunicorn
CMD ["gunicorn", "--bind", "0.0.0.0:5000", "app:app", "--workers", "2", "--threads", "4", "--timeout", "300"]
