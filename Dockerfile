# Dockerfile
FROM python:3.9-slim

WORKDIR /app

# Install system dependencies required for OpenCV and MediaPipe
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better layer caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the ROM library and demo application
COPY rom/ /app/rom/
COPY rom_streamlit_demo.py .

# Expose the ports for the API and Streamlit app
EXPOSE 8000
EXPOSE 8501

# Environment variables
ENV ROM_API_HOST=0.0.0.0
ENV ROM_API_PORT=8000
ENV ROM_LLM_API_URL=http://llm-api:8001/analyze

# Create a startup script
RUN echo '#!/bin/bash\n\
    if [ "$1" = "api" ]; then\n\
    echo "Starting ROM API server..."\n\
    python -m rom.api.main --host ${ROM_API_HOST} --port ${ROM_API_PORT}\n\
    elif [ "$1" = "streamlit" ]; then\n\
    echo "Starting Streamlit demo..."\n\
    streamlit run rom_streamlit_demo.py --server.port=8501 --server.address=0.0.0.0\n\
    else\n\
    echo "Unknown command: $1"\n\
    echo "Usage: docker run ... [api|streamlit]"\n\
    exit 1\n\
    fi' > /app/entrypoint.sh \
    && chmod +x /app/entrypoint.sh

ENTRYPOINT ["/app/entrypoint.sh"]
CMD ["api"]

# # docker-compose.yml
# version: '3.8'

# services:
# rom-api:
# build: .
# command: api
# ports:
# - "8000:8000"
# environment:
# - ROM_DEBUG=False
# - ROM_LOG_LEVEL=INFO
# - ROM_API_HOST=0.0.0.0
# - ROM_API_PORT=8000
# - ROM_LLM_API_URL=http://llm-api:8001/analyze
# volumes:
# - ./logs:/app/logs
# restart: unless-stopped
# networks:
# - rom-network

# rom-streamlit:
# build: .
# command: streamlit
# ports:
# - "8501:8501"
# environment:
# - ROM_API_URL=http://rom-api:8000
# depends_on:
# - rom-api
# restart: unless-stopped
# networks:
# - rom-network

# llm-api:
# image: llm-api:latest  # Replace with your actual LLM API image
# ports:
# - "8001:8001"
# environment:
# - MODEL_PATH=/models/your_model.bin  # Adjust as needed
# volumes:
# - ./models:/models
# restart: unless-stopped
# networks:
# - rom-network

# networks:
# rom-network:
# driver: bridge

# # .dockerignore
# __pycache__/
# *.py[cod]
# *$py.class
# *.so
# .Python
# env/
# build/
# develop-eggs/
# dist/
# downloads/
# eggs/
# .eggs/
# lib/
# lib64/
# parts/
# sdist/
# var/
# *.egg-info/
# .installed.cfg
# *.egg
# .env
# venv/
# .venv/
# .git/
# .github/
# .gitignore
# .gitlab-ci.yml
# Dockerfile
# docker-compose.yml
# .dockerignore
# *.md
# logs/
# data/
# tests/
# examples/
# docs/

# # deployment/kubernetes/rom-deployment.yaml
# apiVersion: apps/v1
# kind: Deployment
# metadata:
# name: rom-api
# labels:
# app: rom-api
# spec:
# replicas: 3
# selector:
# matchLabels:
# app: rom-api
# template:
# metadata:
# labels:
# app: rom-api
# spec:
# containers:
# - name: rom-api
# image: rom-assessment:latest
# args: ["api"]
# ports:
# - containerPort: 8000
# env:
# - name: ROM_API_HOST
# value: "0.0.0.0"
# - name: ROM_API_PORT
# value: "8000"
# - name: ROM_LLM_API_URL
# value: "http://llm-api-service:8001/analyze"
# resources:
# limits:
# cpu: "1"
# memory: "1Gi"
# requests:
# cpu: "500m"
# memory: "512Mi"
# livenessProbe:
# httpGet:
# path: /api/health
# port: 8000
# initialDelaySeconds: 30
# periodSeconds: 10
# readinessProbe:
# httpGet:
# path: /api/health
# port: 8000
# initialDelaySeconds: 5
# periodSeconds: 5
# ---
# apiVersion: apps/v1
# kind: Deployment
# metadata:
# name: rom-streamlit
# labels:
# app: rom-streamlit
# spec:
# replicas: 1
# selector:
# matchLabels:
# app: rom-streamlit
# template:
# metadata:
# labels:
# app: rom-streamlit
# spec:
# containers:
# - name: rom-streamlit
# image: rom-assessment:latest
# args: ["streamlit"]
# ports:
# - containerPort: 8501
# env:
# - name: ROM_API_URL
# value: "http://rom-api-service:8000"
# resources:
# limits:
# cpu: "1"
# memory: "1Gi"
# requests:
# cpu: "500m"
# memory: "512Mi"
# ---
# apiVersion: v1
# kind: Service
# metadata:
# name: rom-api-service
# spec:
# selector:
# app: rom-api
# ports:
# - port: 8000
# targetPort: 8000
# type: ClusterIP
# ---
# apiVersion: v1
# kind: Service
# metadata:
# name: rom-streamlit-service
# spec:
# selector:
# app: rom-streamlit
# ports:
# - port: 8501
# targetPort: 8501
# type: LoadBalancer
# ---
# apiVersion: networking.k8s.io/v1
# kind: Ingress
# metadata:
# name: rom-ingress
# annotations:
# nginx.ingress.kubernetes.io/rewrite-target: /
# spec:
# rules:
# - host: rom.yourdomain.com
# http:
# paths:
# - path: /api
# pathType: Prefix
# backend:
# service:
# name: rom-api-service
# port:
# number: 8000
# - path: /
# pathType: Prefix
# backend:
# service:
# name: rom-streamlit-service
# port:
# number: 8501