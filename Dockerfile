FROM python:3.10-slim

WORKDIR /app

# Install git (needed for pip to pull from GitHub)
RUN apt-get update && \
    apt-get install -y git && \
    rm -rf /var/lib/apt/lists/*

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8501
CMD ["streamlit", "run", "Agent 3: InfoGuide/src/run_updated.py", "--server.port=8501", "--server.enableCORS=false", "--server.baseUrlPath=smartpilot"]
	

